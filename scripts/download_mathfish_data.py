"""
Download MathFish and Achieve-the-Core data from HuggingFace.
No API key or login needed for these public datasets.

Saves to data/raw/ so preprocess_mathfish and build_curriculum_graph can use local paths.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "data", "raw")
MATHFISH_HF = "allenai/mathfish"
ATC_HF = "allenai/achieve-the-core"


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    mathfish_dir = os.path.join(OUT_DIR, "mathfish")
    atc_dir = os.path.join(OUT_DIR, "achieve-the-core")
    os.makedirs(mathfish_dir, exist_ok=True)
    os.makedirs(atc_dir, exist_ok=True)

    print("Downloading MathFish (problems) from HuggingFace...")
    import json
    for split in ("train", "dev", "test"):
        ds = load_dataset(MATHFISH_HF, split=split)
        path = os.path.join(mathfish_dir, f"{split}.jsonl")
        with open(path, "w") as f:
            for row in ds:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        print(f"  Wrote {path} ({len(ds)} rows)")

    print("Downloading Achieve-the-Core (standards.jsonl + domain_groups.json)...")
    for fname in ("standards.jsonl", "domain_groups.json"):
        out_path = os.path.join(atc_dir, fname)
        try:
            hf_hub_download(
                repo_id=ATC_HF,
                filename=fname,
                repo_type="dataset",
                local_dir=atc_dir,
                local_dir_use_symlinks=False,
            )
            print(f"  Downloaded {fname} -> {atc_dir}")
        except Exception as e:
            print(f"  Failed to download {fname}: {e}")
            if fname == "standards.jsonl":
                # Fallback: load as dataset and write jsonl
                atc = load_dataset(ATC_HF, split="train")
                with open(out_path, "w") as f:
                    for row in atc:
                        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                print(f"  Wrote {out_path} from load_dataset ({len(atc)} rows)")

    print("Done. Use these paths:")
    print("  Problems (train):", os.path.join(mathfish_dir, "train.jsonl"))
    print("  Problems (dev):  ", os.path.join(mathfish_dir, "dev.jsonl"))
    print("  Problems (test): ", os.path.join(mathfish_dir, "test.jsonl"))
    print("  Standards:       ", os.path.join(atc_dir, "standards.jsonl"))
    print("  Domain groups:   ", os.path.join(atc_dir, "domain_groups.json"))


if __name__ == "__main__":
    main()
