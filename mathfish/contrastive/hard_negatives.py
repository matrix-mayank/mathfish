"""
Hard negative sampler: 40% siblings, 30% conceptual, 20% grade-adjacent, 10% random.
"""
import random
from typing import List, Set

from .curriculum_graph import CurriculumGraph


class HardNegativeSampler:
    """
    Samples hard negatives for a (problem, positive_standard) pair.
    Ratios: siblings 40%, conceptually connected 30%, grade-adjacent 20%, random 10%.
    """

    def __init__(
        self,
        graph: CurriculumGraph,
        seed: int = 42,
        ratio_sibling: float = 0.40,
        ratio_conceptual: float = 0.30,
        ratio_grade_adjacent: float = 0.20,
        ratio_random: float = 0.10,
    ):
        self.graph = graph
        self.rng = random.Random(seed)
        self.ratio_sibling = ratio_sibling
        self.ratio_conceptual = ratio_conceptual
        self.ratio_grade_adjacent = ratio_grade_adjacent
        self.ratio_random = ratio_random
        self._all_standards = list(graph.standard_ids)

    def sample(
        self,
        positive_standard: str,
        positive_set: Set[str] | None = None,
        num_negatives: int = 5,
    ) -> List[str]:
        """
        Return a list of hard negative standard IDs (no duplicates, not in positive_set).
        positive_set: all gold standards for this problem (to exclude from negatives).
        """
        positive_set = positive_set or {positive_standard}
        exclude = set(positive_set)

        sibs = self.graph.get_siblings(positive_standard, exclude=exclude)
        conceptual = self.graph.get_conceptual_neighbors(positive_standard, exclude=exclude)
        grade_adj = self.graph.get_grade_adjacent_standards(positive_standard, exclude=exclude)

        pool_random = [s for s in self._all_standards if s not in exclude]
        self.rng.shuffle(pool_random)

        n_sib = max(0, int(num_negatives * self.ratio_sibling))
        n_con = max(0, int(num_negatives * self.ratio_conceptual))
        n_gr = max(0, int(num_negatives * self.ratio_grade_adjacent))
        n_rand = max(0, num_negatives - n_sib - n_con - n_gr)

        out: List[str] = []
        used: Set[str] = set()

        def add_from(lst: List[str], n: int) -> None:
            self.rng.shuffle(lst)
            for s in lst:
                if len(out) >= num_negatives:
                    return
                if s not in used:
                    out.append(s)
                    used.add(s)

        add_from(sibs, n_sib)
        add_from(conceptual, n_con)
        add_from(grade_adj, n_gr)
        for s in pool_random:
            if len(out) >= num_negatives:
                break
            if s not in used:
                out.append(s)
                used.add(s)

        return out[:num_negatives]
