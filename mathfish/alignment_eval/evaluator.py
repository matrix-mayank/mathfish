"""
AlignmentEvaluator: runs all metrics given (pred_sets, gold_sets) and optional graph/retriever.
"""
from typing import Callable, List, Set

from .metrics import (
    exact_match_accuracy,
    micro_f1,
    macro_f1,
    recall_at_k,
    avg_graph_distance,
    sibling_confusion_rate,
)


class AlignmentEvaluator:
    """
    Evaluates alignment predictions against gold. Optionally uses a curriculum graph
    for avg_graph_distance and sibling_confusion_rate.
    """

    def __init__(
        self,
        get_distance: Callable[[str, str], int] | None = None,
        get_siblings: Callable[[str], List[str]] | None = None,
    ):
        self.get_distance = get_distance
        self.get_siblings = get_siblings

    def evaluate(
        self,
        pred_sets: List[Set[str]],
        gold_sets: List[Set[str]],
        pred_ranked_lists: List[List[str]] | None = None,
    ) -> dict:
        """
        pred_sets / gold_sets: list of sets of standard IDs (one per problem).
        pred_ranked_lists: optional list of ranked prediction lists (for Recall@k). If None, use pred_sets as list order.
        """
        out = {
            "exact_match": exact_match_accuracy(pred_sets, gold_sets),
            "micro_f1": micro_f1(pred_sets, gold_sets),
            "macro_f1": macro_f1(pred_sets, gold_sets),
        }
        ranked = pred_ranked_lists if pred_ranked_lists is not None else [list(p) for p in pred_sets]
        for k in (5, 10, 20):
            out[f"recall@{k}"] = recall_at_k(ranked, gold_sets, k)
        if self.get_distance is not None:
            out["avg_graph_distance"] = avg_graph_distance(pred_sets, gold_sets, self.get_distance)
        if self.get_siblings is not None:
            out["sibling_confusion_rate"] = sibling_confusion_rate(
                pred_sets, gold_sets, self.get_siblings
            )
        return out
