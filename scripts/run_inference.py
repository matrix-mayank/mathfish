"""
Run bi-encoder inference: embed problems, similarity to all standards, output top-k as predicted standard_ids.
Produces a JSONL that evaluate_alignment.py can use.

Usage:
  python scripts/run_inference.py --checkpoint_dir outputs/biencoder/checkpoint \\
    --problems_file data/processed/problems_dev100.jsonl \\
    --from_hf --output_file data/processed/predictions_biencoder.jsonl --top_k 5
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
        hf_hub_download(repo_id="allenai/achieve-the-core", filename=fname, repo_type="dataset", local_dir=cache_dir)
    return standards_path, domain_groups_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default=os.path.join(ROOT, "outputs", "biencoder", "checkpoint"))
    p.add_argument("--problems_file", type=str, required=True)
    p.add_argument("--from_hf", action="store_true")
    p.add_argument("--standards_path", type=str, default=None)
    p.add_argument("--domain_groups_path", type=str, default=None)
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--top_k", type=int, default=5, help="Number of standards to predict per problem")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--proj_dim", type=int, default=256)
    args = p.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standards: id -> description (for encoding)
    if args.from_hf:
        cache_dir = os.path.join(ROOT, "data", "cache", "achieve-the-core")
        standards_path, domain_groups_path = _load_atc_from_hf(cache_dir)
    else:
        standards_path = args.standards_path
        domain_groups_path = args.domain_groups_path
    if not standards_path or not os.path.isfile(standards_path):
        raise SystemExit("Need --standards_path or --from_hf")
    standard_ids = []
    standard_texts = []
    with open(standards_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("level") != "Standard":
                continue
            standard_ids.append(d["id"])
            standard_texts.append(d.get("description", ""))
    print("Standards:", len(standard_ids))

    # Load checkpoint
    ckpt_dir = args.checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    encoder = AutoModel.from_pretrained(ckpt_dir)
    hidden_size = encoder.config.hidden_size
    proj_problem = torch.nn.Linear(hidden_size, args.proj_dim)
    proj_standard = torch.nn.Linear(hidden_size, args.proj_dim)
    ckpt = torch.load(os.path.join(ckpt_dir, "last.pt"), map_location=device)
    encoder.load_state_dict(ckpt["encoder_state"], strict=False)
    proj_problem.load_state_dict(ckpt["proj_problem_state"])
    proj_standard.load_state_dict(ckpt["proj_standard_state"])
    encoder.to(device)
    proj_problem.to(device)
    proj_standard.to(device)
    encoder.eval()
    proj_problem.eval()
    proj_standard.eval()

    def encode(texts, proj, max_len=256):
        inp = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            out = encoder(**inp)
        mask = inp["attention_mask"]
        h = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
        return torch.nn.functional.normalize(proj(h), p=2, dim=-1)

    # Precompute standard embeddings (batch to avoid OOM)
    print("Encoding standards...")
    std_embs = []
    for i in range(0, len(standard_texts), args.batch_size):
        batch = standard_texts[i : i + args.batch_size]
        std_embs.append(encode(batch, proj_standard))
    std_emb = torch.cat(std_embs, dim=0)  # (N_std, proj_dim)

    # Load problems
    problems = []
    with open(args.problems_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    print("Problems:", len(problems))

    # Encode problems, then similarity -> top-k
    out_path = args.output_file or (args.problems_file.replace(".jsonl", "_predictions.jsonl"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fout:
        for i in range(0, len(problems), args.batch_size):
            batch = problems[i : i + args.batch_size]
            texts = [p["text"] for p in batch]
            q = encode(texts, proj_problem)  # (B, proj_dim)
            sim = q @ std_emb.T  # (B, N_std)
            topk = sim.topk(min(args.top_k, sim.size(1)), dim=1).indices  # (B, top_k)
            for p, row in zip(batch, topk):
                pred_ids = [standard_ids[j] for j in row.cpu().tolist()]
                fout.write(json.dumps({"id": p["id"], "standard_ids": pred_ids}) + "\n")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
