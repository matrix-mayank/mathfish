"""InfoNCE loss for contrastive (problem, positive, negatives)."""
import torch
import torch.nn.functional as F


def infonce_loss(
    problem_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    problem_emb: (B, D), positive_emb: (B, D), negative_emb: (B, K, D).
    Returns scalar: -log( exp(sim(q,p+)/tau) / (exp(sim(q,p+)/tau) + sum_k exp(sim(q,p-_k)/tau)) ).
    All embeddings assumed L2-normalized; sim = dot product.
    """
    # (B,) positive logits
    pos_logits = (problem_emb * positive_emb).sum(dim=-1) / temperature
    # (B, K) negative logits
    neg_logits = torch.bmm(
        problem_emb.unsqueeze(1),
        negative_emb.transpose(1, 2),
    ).squeeze(1) / temperature
    # (B, 1+K) logits; index 0 is positive
    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
    labels = torch.zeros(problem_emb.size(0), dtype=torch.long, device=problem_emb.device)
    return F.cross_entropy(logits, labels)
