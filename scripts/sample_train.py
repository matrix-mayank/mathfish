"""
Sample N examples from MathFish train set (HuggingFace). Reads train.jsonl line-by-line
to avoid HF schema issues. Use for a smaller, faster training run.

Example: python scripts/sample_train.py --n 500
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEED = 42
DEFAULT_N = 500
HF_REPO = "allenai/mathfish"


def clean_text(text: str, elements: dict) -> str:
    import re
    if not text:
        return ""
    for key in (elements or {}):
        if "IMAGE" in key:
            text = text.replace(key, "[IMAGE]")
        elif "TABLE" in key:
            text = text.replace(key, "[TABLE]")
    return re.sub(r"\s+", " ", text).strip()


def load_train_jsonl():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_REPO, filename="train.jsonl", repo_type="dataset")
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    print("Loading train.jsonl from HF (line-by-line)...")
    rows = load_train_jsonl()
    rng = __import__("random").Random(args.seed)
    n = min(args.n, len(rows))
    indices = rng.sample(range(len(rows)), n)
    out_path = args.output or os.path.join(ROOT, "data", "processed", f"problems_train{args.n}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out = []
    skipped = 0
    for i in indices:
        row = rows[i]
        pid = row.get("id") or row.get("problem_id") or f"train_{i}"
        text = row.get("problem_activity") or row.get("content") or row.get("text") or ""
        elements = row.get("elements") or {}
        if isinstance(elements, str):
            try:
                elements = json.loads(elements)
            except Exception:
                elements = {}
        standards = row.get("standards") or []
        # Extract standard IDs and filter for Addressing/Alignment relations
        standard_ids = []
        for s in standards:
            if isinstance(s, (list, tuple)) and len(s) > 1:
                relation, std_id = s[0], s[1]
                if relation in ["Addressing", "Alignment"]:
                    standard_ids.append(std_id)
            elif isinstance(s, str):
                standard_ids.append(s)
        
        # Skip problems without standards
        if not standard_ids:
            skipped += 1
            continue
            
        text = clean_text(text, elements)
        out.append({"id": pid, "text": text, "standard_ids": standard_ids})

    with open(out_path, "w") as f:
        for d in out:
            f.write(json.dumps(d) + "\n")
    print(f"Sampled {len(out)} examples with standards (skipped {skipped} without standards, seed={args.seed}) -> {out_path}")


if __name__ == "__main__":
    main()
