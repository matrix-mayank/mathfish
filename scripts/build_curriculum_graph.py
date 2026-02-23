"""
Build curriculum graph from ATC standards and save to data/processed/curriculum_graph.json.
Uses mathfish.contrastive.CurriculumGraph (which wraps ATCMap + TreeRetriever).

Can load from HuggingFace directly with --from_hf (no separate download step).
"""
import argparse
import json
import os
import sys

# project root = parent of scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mathfish.contrastive import CurriculumGraph

ATC_HF = "allenai/achieve-the-core"


def _load_atc_from_hf(cache_dir: str) -> tuple[str, str]:
    """Load achieve-the-core from HuggingFace; save to cache_dir. Returns (standards_path, domain_groups_path)."""
    os.makedirs(cache_dir, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Install: pip install huggingface_hub")
    standards_path = os.path.join(cache_dir, "standards.jsonl")
    domain_groups_path = os.path.join(cache_dir, "domain_groups.json")
    if os.path.isfile(standards_path) and os.path.isfile(domain_groups_path):
        print("Using cached ATC files in", cache_dir)
        return standards_path, domain_groups_path
    print("Loading achieve-the-core from HuggingFace (cached to", cache_dir, ")...")
    hf_hub_download(
        repo_id=ATC_HF,
        filename="standards.jsonl",
        repo_type="dataset",
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    hf_hub_download(
        repo_id=ATC_HF,
        filename="domain_groups.json",
        repo_type="dataset",
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    return standards_path, domain_groups_path


def parse_args():
    p = argparse.ArgumentParser(description="Build and save curriculum graph from ATC standards.")
    p.add_argument(
        "--from_hf",
        action="store_true",
        help="Load standards + domain_groups from HuggingFace (allenai/achieve-the-core); no local paths needed.",
    )
    p.add_argument(
        "--standards_path",
        type=str,
        default=None,
        help="Path to standards.jsonl (required if not --from_hf).",
    )
    p.add_argument(
        "--domain_groups_path",
        type=str,
        default=None,
        help="Path to domain_groups.json (required if not --from_hf).",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to cache ATC when using --from_hf. Default: data/cache/achieve-the-core",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path. Default: data/processed/curriculum_graph.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.from_hf:
        cache_dir = args.cache_dir or os.path.join(ROOT, "data", "cache", "achieve-the-core")
        standards_path, domain_groups_path = _load_atc_from_hf(cache_dir)
    else:
        if not args.standards_path or not args.domain_groups_path:
            raise SystemExit("Provide --standards_path and --domain_groups_path, or use --from_hf")
        standards_path = args.standards_path
        domain_groups_path = args.domain_groups_path

    output_path = args.output_path or os.path.join(ROOT, "data", "processed", "curriculum_graph.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Building curriculum graph...")
    graph = CurriculumGraph(standards_path, domain_groups_path)
    graph.save(output_path)
    print(f"Saved to {output_path}")
    print(f"Total standard-level IDs: {len(graph.standard_ids)}")


if __name__ == "__main__":
    main()
