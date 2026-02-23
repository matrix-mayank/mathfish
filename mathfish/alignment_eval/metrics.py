"""
Alignment metrics: exact match, micro/macro F1, Recall@k, avg graph distance, sibling confusion.
"""
from typing import Dict, List, Set, Tuple

import numpy as np


def exact_match_accuracy(pred_sets: List[Set[str]], gold_sets: List[Set[str]]) -> float:
    """Fraction of problems where predicted set exactly equals gold set."""
    if not pred_sets:
        return 0.0
    correct = sum(1 for p, g in zip(pred_sets, gold_sets) if set(p) == set(g))
    return correct / len(pred_sets)


def _per_problem_f1(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    """Precision, recall, F1 for one problem. Returns (p, r, f1); f1=0 if both p and r are 0."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0, 0.0
    inter = len(pred & gold)
    p = inter / len(pred)
    r = inter / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def micro_f1(pred_sets: List[Set[str]], gold_sets: List[Set[str]]) -> float:
    """Micro-averaged F1: total TP, FP, FN then F1."""
    tp = fp = fn = 0
    for pred, gold in zip(pred_sets, gold_sets):
        inter = len(pred & gold)
        tp += inter
        fp += len(pred - gold)
        fn += len(gold - pred)
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def macro_f1(pred_sets: List[Set[str]], gold_sets: List[Set[str]]) -> float:
    """Macro-averaged F1: mean of per-problem F1."""
    if not pred_sets:
        return 0.0
    f1s = [_per_problem_f1(p, g)[2] for p, g in zip(pred_sets, gold_sets)]
    return float(np.mean(f1s))


def recall_at_k(
    pred_ranked_lists: List[List[str]],
    gold_sets: List[Set[str]],
    k: int,
) -> float:
    """Recall@k: for each problem, is at least one gold in the top-k predictions? Mean over problems."""
    if not pred_ranked_lists:
        return 0.0
    recalls = []
    for pred_list, gold in zip(pred_ranked_lists, gold_sets):
        top_k = set(pred_list[:k])
        if not gold:
            recalls.append(1.0)
        else:
            recalls.append(len(top_k & gold) / len(gold))
    return float(np.mean(recalls))


def avg_graph_distance(
    pred_sets: List[Set[str]],
    gold_sets: List[Set[str]],
    get_distance,
) -> float:
    """
    For each predicted standard, min distance to any gold standard; average over all predictions.
    get_distance(standard_id_a, standard_id_b) -> int (path length). Uses NetworkX no-path as exception.
    """
    import networkx as nx

    distances = []
    for pred_set, gold_set in zip(pred_sets, gold_sets):
        if not pred_set:
            continue
        if not gold_set:
            # no gold: skip or use a default; we skip to match paper-style metric
            continue
        for p in pred_set:
            min_d = float("inf")
            for g in gold_set:
                try:
                    d = get_distance(p, g)
                    min_d = min(min_d, d)
                except (nx.NetworkXNoPath, Exception):
                    pass
            if min_d != float("inf"):
                distances.append(min_d)
    return float(np.mean(distances)) if distances else float("nan")


def sibling_confusion_rate(
    pred_sets: List[Set[str]],
    gold_sets: List[Set[str]],
    get_siblings,
) -> float:
    """
    Among wrong predictions: fraction where the predicted standard is a sibling of a gold standard.
    get_siblings(standard_id) -> list of sibling standard ids.
    """
    wrong_preds = 0
    sibling_wrong = 0
    for pred_set, gold_set in zip(pred_sets, gold_sets):
        wrong = pred_set - gold_set
        if not wrong:
            continue
        for p in wrong:
            wrong_preds += 1
            sibs = set(get_siblings(p))
            if gold_set & sibs:
                sibling_wrong += 1
    return (sibling_wrong / wrong_preds) if wrong_preds else 0.0
