"""
Curriculum graph for hard negative sampling.
Builds from ATC standards: hierarchy (cluster = parent of standard) + conceptual connections.
Exposes siblings (same cluster), conceptual neighbors (ATC graph), grade-adjacent standards.
"""
from collections import defaultdict
import json
from typing import List, Set

from mathfish.tree_retriever import ATCMap, TreeRetriever
from mathfish.utils import get_grade, map_grade_to_number, map_number_to_grade


class CurriculumGraph:
    """
    Wraps ATC hierarchy and coherence map for curriculum-aware negative sampling.
    """

    def __init__(self, standards_path: str, domain_groups_path: str):
        self.standards_path = standards_path
        self.domain_groups_path = domain_groups_path
        self.atc = ATCMap(standards_path)
        self.atc.create_undirected_graph()
        self.retriever = TreeRetriever(standards_path, domain_groups_path)

        # standard_id -> cluster (parent) for Standard-level only
        self.standard_to_cluster: dict[str, str] = {}
        # standard_id -> grade string (K, 1, ..., 8, HS)
        self.standard_to_grade: dict[str, str] = {}
        # cluster -> set of standard ids (siblings)
        self.cluster_to_standards: dict[str, Set[str]] = defaultdict(set)
        # all standard-level ids
        self.standard_ids: Set[str] = set()

        with open(standards_path, "r") as f:
            for line in f:
                d = json.loads(line)
                if d["level"] != "Standard":
                    continue
                sid = d["id"]
                self.standard_ids.add(sid)
                parent = d.get("parent") or ""
                self.standard_to_cluster[sid] = parent
                self.standard_to_grade[sid] = get_grade(sid)
                if parent:
                    self.cluster_to_standards[parent].add(sid)

        self._grade_to_standards: dict[str, Set[str]] = defaultdict(set)
        for sid in self.standard_ids:
            self._grade_to_standards[self.standard_to_grade[sid]].add(sid)

    def get_siblings(self, standard_id: str, exclude: Set[str] | None = None) -> List[str]:
        """Same-cluster (sibling) standards. Exclude set can be positive standards."""
        exclude = exclude or set()
        cluster = self.standard_to_cluster.get(standard_id)
        if not cluster:
            return []
        sibs = self.cluster_to_standards.get(cluster, set()) - {standard_id} - exclude
        return list(sibs)

    def get_conceptual_neighbors(self, standard_id: str, exclude: Set[str] | None = None) -> List[str]:
        """Standards connected in ATC coherence map (any relation type)."""
        exclude = exclude or set()
        if standard_id not in self.atc.connections:
            return []
        neighbors = set()
        for _rel, dest_list in self.atc.connections[standard_id].items():
            for d in dest_list:
                if d in self.standard_ids and d != standard_id:
                    neighbors.add(d)
        return list(neighbors - exclude)

    def get_grade_adjacent_standards(
        self, standard_id: str, exclude: Set[str] | None = None, delta: int = 1
    ) -> List[str]:
        """Standards in grade ± delta (K=0, 1..8, HS=9)."""
        exclude = exclude or set()
        g = self.standard_to_grade.get(standard_id)
        if not g:
            return []
        num = map_grade_to_number(g)
        adjacent_nums = [num + d for d in (-delta, delta) if 0 <= num + d <= 9]
        out = set()
        for n in adjacent_nums:
            grade_str = map_number_to_grade(n)
            if grade_str and grade_str in self._grade_to_standards:
                out |= self._grade_to_standards[grade_str]
        out -= {standard_id}
        out -= exclude
        return list(out)

    def get_distance(self, start: str, end: str) -> int:
        """Shortest path length in ATC undirected graph. Raises if no path."""
        return self.atc.get_distance(start, end, directed=False)

    def save(self, path: str) -> None:
        """Persist graph indices to JSON for fast reload (optional)."""
        import networkx as nx
        data = {
            "standard_to_cluster": self.standard_to_cluster,
            "standard_to_grade": self.standard_to_grade,
            "cluster_to_standards": {k: list(v) for k, v in self.cluster_to_standards.items()},
            "standard_ids": list(self.standard_ids),
            "grade_to_standards": {k: list(v) for k, v in self._grade_to_standards.items()},
            "edges": list(self.atc.undir_graph.edges()),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=0)
