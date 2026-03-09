"""
Train contrastive bi-encoder (Component 1).
Loads problems jsonl, curriculum graph (--from_hf or paths), builds ContrastiveDataset,
trains with InfoNCE loss. Saves checkpoints to output dir.

Example (after sampling dev and building ATC cache):
  python scripts/train_biencoder.py --problems_file data/processed/problems_dev100.jsonl --from_hf --max_steps 50
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_atc_from_hf(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    standards_path = os.path.join(cache_dir, "standards.jsonl")
    domain_groups_path = os.path.join(cache_dir, "domain_groups.json")
    if os.path.isfile(standards_path) and os.path.isfile(domain_groups_path):
        return standards_path, domain_groups_path
    from huggingface_hub import hf_hub_download
    for fname in ("standards.jsonl", "domain_groups.json"):
        hf_hub_download(repo_id="allenai/achieve-the-core", filename=fname, repo_type="dataset", local_dir=cache_dir, local_dir_use_symlinks=False)
    return standards_path, domain_groups_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problems_file", type=str, default=os.path.join(ROOT, "data", "processed", "problems_dev100.jsonl"))
    p.add_argument("--from_hf", action="store_true", help="Load ATC from HuggingFace (cache to data/cache/achieve-the-core)")
    p.add_argument("--standards_path", type=str, default=None)
    p.add_argument("--domain_groups_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=os.path.join(ROOT, "outputs", "biencoder"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=100, help="Max optimization steps (for quick runs)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--num_negatives", type=int, default=4)
    p.add_argument("--max_len", type=int, default=256, help="Max token length for problem/standard text. Lower = faster.")
    p.add_argument("--quick", action="store_true", help="Quick run: 50 steps, max_len=128, batch_size=16.")
    p.add_argument("--fp16", action="store_true", help="Use mixed precision (faster on GPU).")
    p.add_argument("--early_stopping_patience", type=int, default=20, help="Stop when loss has not improved for this many steps (0 = disabled).")
    p.add_argument("--improved", action="store_true", help="Better accuracy: 500 steps, patience 40, 8 negatives, temperature 0.03.")
    args = p.parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 50)
        args.max_len = 128
        args.batch_size = 16
    if args.improved:
        args.max_steps = max(args.max_steps, 500)
        args.early_stopping_patience = 40
        args.num_negatives = 8
        args.temperature = 0.03

    import torch
    from torch.utils.data import DataLoader
    from mathfish.contrastive import CurriculumGraph, HardNegativeSampler, ContrastiveDataset
    from mathfish.contrastive.losses import infonce_loss

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ATC paths
    print("Loading ATC (standards + domain groups)...")
    if args.from_hf:
        cache_dir = os.path.join(ROOT, "data", "cache", "achieve-the-core")
        standards_path, domain_groups_path = _load_atc_from_hf(cache_dir)
    else:
        if not args.standards_path or not args.domain_groups_path:
            raise SystemExit("Provide --standards_path and --domain_groups_path, or use --from_hf")
        standards_path = args.standards_path
        domain_groups_path = args.domain_groups_path
    print("ATC paths OK")

    # Load problems
    print("Loading problems from", args.problems_file, "...")
    if not os.path.isfile(args.problems_file):
        raise SystemExit(f"Problems file not found: {args.problems_file}")
    problems = []
    with open(args.problems_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    problem_texts = [x["text"] for x in problems]
    problem_standard_ids = [x.get("standard_ids", []) for x in problems]
    # Filter to problems that have at least one standard we can use
    print("Building curriculum graph...")
    graph = CurriculumGraph(standards_path, domain_groups_path)
    standard_id_to_text = dict(graph.retriever.standards_descriptions)
    sampler = HardNegativeSampler(graph, seed=args.seed, ratio_sibling=0.4, ratio_conceptual=0.3, ratio_grade_adjacent=0.2, ratio_random=0.1)

    # Keep only problems with at least one standard in our id_to_text
    valid = []
    for i, (text, std_ids) in enumerate(zip(problem_texts, problem_standard_ids)):
        if any(sid in standard_id_to_text for sid in std_ids):
            valid.append(i)
    if not valid:
        raise SystemExit("No problems with at least one known standard. Check standards_path / ATC.")
    problem_texts = [problem_texts[i] for i in valid]
    problem_standard_ids = [problem_standard_ids[i] for i in valid]
    print(f"Using {len(problem_texts)} problems with known standards")

    print("Building contrastive dataset...")
    dataset = ContrastiveDataset(
        problem_texts=problem_texts,
        problem_standard_ids=problem_standard_ids,
        standard_id_to_text=standard_id_to_text,
        hard_negative_sampler=sampler,
        num_negatives=args.num_negatives,
        max_negatives=args.num_negatives,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    print("Dataset size:", len(dataset))

    # Model: HF encoder + projection (trainable)
    print("Loading encoder and tokenizer:", args.model_name, "...")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)
    hidden_size = encoder.config.hidden_size
    proj_problem = torch.nn.Linear(hidden_size, args.proj_dim)
    proj_standard = torch.nn.Linear(hidden_size, args.proj_dim)
    encoder.to(device)
    proj_problem.to(device)
    proj_standard.to(device)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj_problem.parameters()) + list(proj_standard.parameters()),
        lr=args.lr,
    )

    max_len = getattr(args, "max_len", 256)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def _encode(texts, proj, seq_len=None):
        seq_len = seq_len or max_len
        inp = tokenizer(texts, padding=True, truncation=True, max_length=seq_len, return_tensors="pt").to(device)
        out = encoder(**inp)
        mask = inp["attention_mask"]
        h = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
        return torch.nn.functional.normalize(proj(h), p=2, dim=-1)

    os.makedirs(args.output_dir, exist_ok=True)
    encoder.train()
    proj_problem.train()
    proj_standard.train()
    print("Starting training (max_steps=%d, early_stopping_patience=%d)..." % (args.max_steps, args.early_stopping_patience), flush=True)
    step = 0
    best_loss = float("inf")
    steps_since_best = 0
    best_state = None  # will hold state_dicts for best step
    
    # Create infinite dataloader iterator
    import itertools
    dataloader_infinite = itertools.cycle(dataloader)
    
    for batch in dataloader_infinite:
        if step >= args.max_steps:
            break
        problems_txt = [x["problem_text"] for x in batch]
        pos_txt = [x["positive_standard_text"] for x in batch]
        neg_txts = [x["negative_standard_texts"] for x in batch]
        K = max(len(n) for n in neg_txts) or 1
        neg_flat = []
        for nlist in neg_txts:
            nlist = (nlist or [""])[:K]
            while len(nlist) < K:
                nlist.append(nlist[0] if nlist else "")
            neg_flat.extend(nlist)
        B = len(batch)
        if use_amp:
            with torch.cuda.amp.autocast():
                q = _encode(problems_txt, proj_problem)
                p_pos = _encode(pos_txt, proj_standard)
                p_neg = _encode(neg_flat, proj_standard)
                p_neg = p_neg.view(B, K, -1)
                loss = infonce_loss(q, p_pos, p_neg, temperature=args.temperature)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            q = _encode(problems_txt, proj_problem)
            p_pos = _encode(pos_txt, proj_standard)
            p_neg = _encode(neg_flat, proj_standard)
            p_neg = p_neg.view(B, K, -1)
            loss = infonce_loss(q, p_pos, p_neg, temperature=args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
        step += 1
        loss_val = loss.item()
        print(f"step {step} loss {loss_val:.4f}", flush=True)
        if loss_val < best_loss:
            best_loss = loss_val
            steps_since_best = 0
            best_state = {
                "encoder_state": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "proj_problem_state": {k: v.cpu().clone() for k, v in proj_problem.state_dict().items()},
                "proj_standard_state": {k: v.cpu().clone() for k, v in proj_standard.state_dict().items()},
                "step": step,
            }
        else:
            steps_since_best += 1
        if args.early_stopping_patience > 0 and steps_since_best >= args.early_stopping_patience:
            print(f"Early stopping at step {step} (no improvement for {args.early_stopping_patience} steps, best loss {best_loss:.4f})", flush=True)
            break

    # Save best checkpoint (so inference gets the best model, not last)
    ckpt_dir = os.path.join(args.output_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    if best_state is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in best_state["encoder_state"].items()})
        proj_problem.load_state_dict({k: v.to(device) for k, v in best_state["proj_problem_state"].items()})
        proj_standard.load_state_dict({k: v.to(device) for k, v in best_state["proj_standard_state"].items()})
        step = best_state["step"]
        print(f"Restored best checkpoint from step {step} (loss {best_loss:.4f})", flush=True)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "proj_problem_state": proj_problem.state_dict(),
        "proj_standard_state": proj_standard.state_dict(),
        "step": step,
    }, os.path.join(ckpt_dir, "last.pt"))
    tokenizer.save_pretrained(ckpt_dir)
    encoder.save_pretrained(ckpt_dir)
    print(f"Saved to {ckpt_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("ERROR:", e)
        traceback.print_exc()
        sys.exit(1)
