"""
Bi-encoder: sentence transformer backbone + 256-d projection, L2-normalized embeddings.
Used for problem and standard encoding in contrastive learning.
"""
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class BiEncoder(nn.Module):
    """
    Shared backbone (e.g. all-mpnet-base-v2) with optional separate projection heads
    for problem and standard, outputting 256-d L2-normalized vectors.
    """

    def __init__(
        self,
        backbone_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        proj_dim: int = 256,
        use_separate_heads: bool = True,
    ):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        self.backbone = SentenceTransformer(backbone_name)
        hidden_size = self.backbone.get_sentence_embedding_dimension()
        self.proj_dim = proj_dim
        self.use_separate_heads = use_separate_heads
        self.proj_problem = nn.Linear(hidden_size, proj_dim)
        if use_separate_heads:
            self.proj_standard = nn.Linear(hidden_size, proj_dim)
        else:
            self.proj_standard = self.proj_problem

    def _encode_with_backbone(self, texts: List[str], device) -> torch.Tensor:
        with torch.no_grad() if not self.training else torch.enable_grad():
            emb = self.backbone.encode(
                texts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )
        return emb

    def forward_problem(self, problem_emb: torch.Tensor) -> torch.Tensor:
        """problem_emb: (batch, hidden_size) from backbone."""
        x = self.proj_problem(problem_emb)
        return F.normalize(x, p=2, dim=-1)

    def forward_standard(self, standard_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj_standard(standard_emb)
        return F.normalize(x, p=2, dim=-1)

    def encode_problems(self, texts: List[str], device) -> torch.Tensor:
        emb = self._encode_with_backbone(texts, device)
        return self.forward_problem(emb)

    def encode_standards(self, texts: List[str], device) -> torch.Tensor:
        emb = self._encode_with_backbone(texts, device)
        return self.forward_standard(emb)
