"""
PyTorch Dataset for contrastive training: (problem_text, positive_standard_id, hard_negative_ids).
"""
from typing import List, Optional

import torch
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    """
    Each item: problem text, one positive standard id, list of hard negative standard ids.
    Used with a tokenizer/collator that turns (text, standard_texts) into input_ids.
    """

    def __init__(
        self,
        problem_texts: List[str],
        problem_standard_ids: List[List[str]],
        standard_id_to_text: dict,
        hard_negative_sampler,
        num_negatives: int = 5,
        max_negatives: int = 5,
    ):
        """
        problem_texts: list of problem text (one per problem).
        problem_standard_ids: list of list of gold standard IDs per problem.
        standard_id_to_text: mapping standard_id -> description string.
        hard_negative_sampler: HardNegativeSampler instance.
        num_negatives: number of hard negatives to sample per positive.
        max_negatives: cap per example (for batching).
        """
        self.problem_texts = problem_texts
        self.problem_standard_ids = problem_standard_ids
        self.standard_id_to_text = standard_id_to_text
        self.sampler = hard_negative_sampler
        self.num_negatives = num_negatives
        self.max_negatives = max_negatives
        # Flatten to (problem_idx, positive_standard_id) for each positive
        self.examples: List[tuple] = []
        for i, (text, std_ids) in enumerate(zip(problem_texts, problem_standard_ids)):
            for pos_id in std_ids:
                if pos_id in standard_id_to_text:
                    self.examples.append((i, pos_id))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        problem_idx, positive_id = self.examples[idx]
        problem_text = self.problem_texts[problem_idx]
        positive_set = set(self.problem_standard_ids[problem_idx])
        neg_ids = self.sampler.sample(
            positive_id,
            positive_set=positive_set,
            num_negatives=self.num_negatives,
        )[: self.max_negatives]
        return {
            "problem_text": problem_text,
            "positive_standard_id": positive_id,
            "positive_standard_text": self.standard_id_to_text.get(positive_id, ""),
            "negative_standard_ids": neg_ids,
            "negative_standard_texts": [self.standard_id_to_text.get(n, "") for n in neg_ids],
        }
