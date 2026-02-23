"""
Load MathFish problems (from HuggingFace or local jsonl), clean text, standardize standards,
split into train/val/test (80% train+val / 20% test, then 80% train / 20% val of train+val).
Save to data/processed/problems_{train,val,test}.jsonl.
Each line: {"id": str, "text": str, "standard_ids": [str]} (standard_ids = list of ATC standard IDs).
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def clean_problem_text(text: str, elements: dict) -> str:
    """Replace image/table placeholders with a single token; normalize whitespace."""
    if not text:
        return ""
    for key, val in (elements or {}).items():
        if "IMAGE" in key:
            text = text.replace(key, "[IMAGE]")
        elif "TABLE" in key:
            text = text.replace(key, "[TABLE]")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_problems_jsonl(path: str) -> list[dict]:
    """Load problems from a jsonl. Expected keys: id, (problem_activity or content or text), standards, elements."""
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_problems_hf(dataset_name: str = "allenai/mathfish", split: str = "train") -> list[dict]:
    """Load from HuggingFace datasets. Converts to same format as jsonl."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    ds = load_dataset(dataset_name, split=split)
    out = []
    for i, row in enumerate(ds):
        problem_id = row.get("id") or row.get("problem_id") or f"prob_{i}"
        text = row.get("problem_activity") or row.get("content") or row.get("text") or ""
        elements = row.get("elements") or {}
        if isinstance(elements, str):
            try:
                elements = json.loads(elements)
            except Exception:
                elements = {}
        standards = row.get("standards") or []
        # standards may be list of [relation, id] or list of ids
        standard_ids = []
        for s in standards:
            if isinstance(s, (list, tuple)):
                standard_ids.append(s[1] if len(s) > 1 else s[0])
            else:
                standard_ids.append(s)
        text = clean_problem_text(text, elements)
        out.append({"id": problem_id, "text": text, "standard_ids": standard_ids})
    return out


def standardize_standards(problems: list[dict], standardizer) -> None:
    """In-place: map each standard_ids entry to standardized ID via standardizer."""
    for p in problems:
        new_ids = []
        for sid in p.get("standard_ids", []):
            s = standardizer.standardize_single_standard(sid) if hasattr(standardizer, "standardize_single_standard") else sid
            if s:
                new_ids.append(s)
        p["standard_ids"] = new_ids


def main():
    parser = argparse.ArgumentParser(description="Preprocess MathFish problems and split train/val/test.")
    parser.add_argument("--input_file", type=str, default=None, help="Local jsonl of problems (optional).")
    parser.add_argument("--hf_dataset", type=str, default="allenai/mathfish", help="HuggingFace dataset name (used when no --input_file).")
    parser.add_argument("--hf_split", type=str, default=None, help="Single HF split (e.g. train). If unset and using HF, uses --use_hf_splits.")
    parser.add_argument("--use_hf_splits", action="store_true", help="Load train/dev/test from HF and use as-is (paper splits); no random split.")
    parser.add_argument("--standards_path", type=str, default=None, help="Optional: standards.jsonl for standardizing IDs.")
    parser.add_argument("--output_dir", type=str, default=None, help="Default: data/processed")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction for test when not using --use_hf_splits.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of train+val used as val when not using --use_hf_splits.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(ROOT, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    if args.input_file and os.path.isfile(args.input_file):
        problems = load_problems_jsonl(args.input_file)
        # Normalize to {id, text, standard_ids}
        normalized = []
        for d in problems:
            pid = d.get("id", "")
            text = d.get("problem_activity") or d.get("content") or d.get("text") or ""
            elements = d.get("elements") or {}
            standards = d.get("standards") or []
            standard_ids = [t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else t for t in standards]
            text = clean_problem_text(text, elements)
            normalized.append({"id": pid, "text": text, "standard_ids": standard_ids})
        problems = normalized
        train, val, test = None, None, problems  # single split
    elif args.use_hf_splits:
        # Load train, dev, test directly from HuggingFace (paper splits); no local download step
        print("Loading train/dev/test from HuggingFace (no local download)...")
        train = load_problems_hf(args.hf_dataset, "train")
        val = load_problems_hf(args.hf_dataset, "dev")
        test = load_problems_hf(args.hf_dataset, "test")
        problems = None
    else:
        split = args.hf_split or "train"
        print(f"Loading {split} from HuggingFace (no local download)...")
        problems = load_problems_hf(args.hf_dataset, split)
        train, val, test = None, None, None

    if args.standards_path and os.path.isfile(args.standards_path):
        from mathfish.preprocessors import StandardStandardizer
        standardizer = StandardStandardizer(args.standards_path)
        if problems is not None:
            standardize_standards(problems, standardizer)
        else:
            for data in (train, val, test):
                if data is not None:
                    standardize_standards(data, standardizer)

    if train is None:
        # Split: first 20% test, then 80% -> 80% train / 20% val
        import random
        rng = random.Random(args.seed)
        rng.shuffle(problems)
        n = len(problems)
        n_test = int(n * args.test_ratio)
        n_rest = n - n_test
        n_val = int(n_rest * args.val_ratio)
        n_train = n_rest - n_val
        test = problems[:n_test]
        rest = problems[n_test:]
        val = rest[:n_val]
        train = rest[n_val:]

    for name, data in [("train", train), ("val", val), ("test", test)]:
        if data is None:
            continue
        path = os.path.join(output_dir, f"problems_{name}.jsonl")
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
        print(f"Wrote {len(data)} problems to {path}")

    print("Done.")


if __name__ == "__main__":
    main()
