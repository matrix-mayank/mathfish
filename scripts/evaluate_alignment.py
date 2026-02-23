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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_file", type=str, required=True, help="JSONL: id, standard_ids (list)")
    p.add_argument("--gold_file", type=str, default=None, help="Default: data/processed/problems_test.jsonl")
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
    pred = {}
    with open(args.predictions_file) as f:
        for line in f:
            d = json.loads(line)
            pred[d["id"]] = set(d.get("standard_ids", d.get("prediction", [])))
    ids = sorted(gold.keys())
    pred_sets = [pred.get(i, set()) for i in ids]
    gold_sets = [gold[i] for i in ids]

    from mathfish.alignment_eval import AlignmentEvaluator

    get_distance = get_siblings = None
    if args.standards_path and args.domain_groups_path and os.path.isfile(args.standards_path):
        from mathfish.contrastive import CurriculumGraph
        graph = CurriculumGraph(args.standards_path, args.domain_groups_path)
        get_distance = graph.get_distance
        get_siblings = lambda s: graph.get_siblings(s)

    evaluator = AlignmentEvaluator(get_distance=get_distance, get_siblings=get_siblings)
    metrics = evaluator.evaluate(pred_sets, gold_sets)
    print(json.dumps(metrics, indent=2))
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Wrote", args.output_file)


if __name__ == "__main__":
    main()
