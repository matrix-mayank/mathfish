"""
Evaluate alignment predictions: exact match, F1, Recall@k, graph distance, sibling confusion.
Loads test set and (optional) curriculum graph; accepts predictions from a jsonl or from a model pipeline.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_file", type=str, required=True, help="JSONL: id, standard_ids (list)")
    p.add_argument("--gold_file", type=str, default=None, help="Default: data/processed/problems_test.jsonl")
    p.add_argument("--from_hf", action="store_true", help="Load ATC from HuggingFace (for graph distance and sibling confusion)")
    p.add_argument("--standards_path", type=str, default=None, help="For graph metrics")
    p.add_argument("--domain_groups_path", type=str, default=None)
    p.add_argument("--output_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    gold_path = args.gold_file or os.path.join(ROOT, "data", "processed", "problems_test.jsonl")
    gold = {}
    if os.path.isfile(gold_path):
        with open(gold_path) as f:
            for line in f:
                d = json.loads(line)
                gold[d["id"]] = set(d.get("standard_ids", []))
    # Load predictions as ordered lists (not sets) to preserve ranking for Recall@k
    pred_ranked = {}  # id -> ordered list of standard_ids
    with open(args.predictions_file) as f:
        for line in f:
            d = json.loads(line)
            pred_ranked[d["id"]] = d.get("standard_ids", d.get("prediction", []))
    ids = sorted(gold.keys())
    pred_sets = [set(pred_ranked.get(i, [])) for i in ids]
    pred_ranked_lists = [pred_ranked.get(i, []) for i in ids]
    gold_sets = [gold[i] for i in ids]

    from mathfish.alignment_eval import AlignmentEvaluator

    get_distance = get_siblings = None
    # --from_hf: auto-download ATC standards for graph metrics
    if args.from_hf:
        cache_dir = os.path.join(ROOT, "data", "cache", "achieve-the-core")
        args.standards_path, args.domain_groups_path = _load_atc_from_hf(cache_dir)
    if args.standards_path and args.domain_groups_path and os.path.isfile(args.standards_path):
        from mathfish.contrastive import CurriculumGraph
        graph = CurriculumGraph(args.standards_path, args.domain_groups_path)
        get_distance = graph.get_distance
        get_siblings = lambda s: graph.get_siblings(s)

    evaluator = AlignmentEvaluator(get_distance=get_distance, get_siblings=get_siblings)
    metrics = evaluator.evaluate(pred_sets, gold_sets, pred_ranked_lists=pred_ranked_lists)
    print(json.dumps(metrics, indent=2))
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Wrote", args.output_file)


if __name__ == "__main__":
    main()
