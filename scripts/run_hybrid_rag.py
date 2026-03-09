"""
Hybrid RAG: Bi-encoder retrieval + Verifier reranking
Stage 1: Bi-encoder retrieves top-k candidates
Stage 2: Verifier classifies each candidate and filters by threshold
"""
import argparse
import json
import os
import sys
import torch
from tqdm import tqdm

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
        hf_hub_download(repo_id="allenai/achieve-the-core", filename=fname, repo_type="dataset", 
                       local_dir=cache_dir, local_dir_use_symlinks=False)
    return standards_path, domain_groups_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biencoder_path", type=str, default="outputs/biencoder")
    parser.add_argument("--verifier_path", type=str, default="outputs/verifier")
    parser.add_argument("--problems_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=20, help="Retrieve top-k candidates from bi-encoder")
    parser.add_argument("--verification_threshold", type=float, default=0.2, 
                       help="Verifier score threshold (0-1)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load standards
    print("Loading standards from HuggingFace...")
    standards_path, _ = _load_atc_from_hf(os.path.join(ROOT, "data", "cache", "achieve-the-core"))
    standards = {}
    with open(standards_path) as f:
        for line in f:
            obj = json.loads(line)
            standards[obj["id"]] = obj["description"]
    print(f"Loaded {len(standards)} standards")

    # Load bi-encoder
    print(f"Loading bi-encoder from {args.biencoder_path}...")
    from transformers import AutoModel, AutoTokenizer
    
    checkpoint_dir = os.path.join(args.biencoder_path, "checkpoint")
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    model_name = config.get("_name_or_path", "sentence-transformers/all-MiniLM-L6-v2")
    proj_dim = config.get("proj_dim", 256)
    
    tokenizer_bi = AutoTokenizer.from_pretrained(checkpoint_dir)
    encoder = AutoModel.from_pretrained(checkpoint_dir)
    hidden_size = encoder.config.hidden_size
    
    proj_problem = torch.nn.Linear(hidden_size, proj_dim)
    proj_standard = torch.nn.Linear(hidden_size, proj_dim)
    
    # Load checkpoint weights
    ckpt = torch.load(os.path.join(checkpoint_dir, "last.pt"), map_location=device)
    encoder.load_state_dict(ckpt["encoder_state"])
    proj_problem.load_state_dict(ckpt["proj_problem_state"])
    proj_standard.load_state_dict(ckpt["proj_standard_state"])
    
    encoder.to(device).eval()
    proj_problem.to(device).eval()
    proj_standard.to(device).eval()

    def encode_texts(texts, proj, max_len=256):
        with torch.no_grad():
            inp = tokenizer_bi(texts, padding=True, truncation=True, max_length=max_len, 
                             return_tensors="pt").to(device)
            out = encoder(**inp)
            mask = inp["attention_mask"]
            h = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
            return torch.nn.functional.normalize(proj(h), p=2, dim=-1)

    # Encode all standards once
    print("Encoding standards...")
    standard_ids = list(standards.keys())
    standard_texts = [standards[sid] for sid in standard_ids]
    
    batch_size = 32
    standard_embeddings = []
    for i in tqdm(range(0, len(standard_texts), batch_size), desc="Encoding standards"):
        batch = standard_texts[i:i+batch_size]
        embs = encode_texts(batch, proj_standard)
        standard_embeddings.append(embs.cpu())
    standard_embeddings = torch.cat(standard_embeddings, dim=0)

    # Load verifier
    print(f"Loading verifier from {args.verifier_path}...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    tokenizer_ver = AutoTokenizer.from_pretrained(args.verifier_path)
    verifier = AutoModelForSequenceClassification.from_pretrained(args.verifier_path)
    verifier.to(device).eval()

    # Load problems
    print(f"Loading problems from {args.problems_file}...")
    problems = []
    with open(args.problems_file) as f:
        for line in f:
            problems.append(json.loads(line))
    print(f"Loaded {len(problems)} problems")

    # Run hybrid inference
    print(f"Running hybrid RAG (top_k={args.top_k}, threshold={args.verification_threshold})...")
    predictions = []
    
    for prob in tqdm(problems, desc="Processing problems"):
        problem_text = prob.get("text", prob.get("problem", ""))
        if not problem_text:
            continue
        
        # Stage 1: Bi-encoder retrieval
        prob_emb = encode_texts([problem_text], proj_problem)
        scores = (prob_emb @ standard_embeddings.T).squeeze()
        top_k_indices = torch.topk(scores, min(args.top_k, len(standard_ids))).indices.cpu().numpy()
        
        candidates = [standard_ids[idx] for idx in top_k_indices]
        
        # Stage 2: Verifier reranking
        verified = []
        if candidates:
            # Batch verification
            ver_inputs = [f"{problem_text} [SEP] {standards[sid]}" for sid in candidates]
            with torch.no_grad():
                inputs = tokenizer_ver(ver_inputs, padding=True, truncation=True, 
                                      max_length=512, return_tensors="pt").to(device)
                outputs = verifier(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()  # P(aligned)
            
            for sid, prob_score in zip(candidates, probs):
                if prob_score >= args.verification_threshold:
                    verified.append(sid)
            
            # Debug: Print scores for first problem
            if len(predictions) == 0 and args.verification_threshold == 0.5:
                print(f"\n🔍 Debug - Verifier scores for problem 1:")
                for sid, score in zip(candidates[:10], probs[:10]):
                    print(f"  {sid}: {score:.4f}")
        
        predictions.append({
            "id": prob.get("id", prob.get("problem_id", "")),
            "standard_ids": verified
        })
    
    # Save predictions
    print(f"Saving predictions to {args.output_file}...")
    with open(args.output_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    
    # Quick stats
    num_with_preds = sum(1 for p in predictions if p["standard_ids"])
    avg_preds = sum(len(p["standard_ids"]) for p in predictions) / len(predictions)
    print(f"\n✅ Done!")
    print(f"Problems with predictions: {num_with_preds}/{len(predictions)} ({100*num_with_preds/len(predictions):.1f}%)")
    print(f"Average standards per problem: {avg_preds:.2f}")


if __name__ == "__main__":
    main()
