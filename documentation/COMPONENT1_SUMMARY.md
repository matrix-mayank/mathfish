# Component 1: Curriculum Structure–Aware Contrastive Learning — Summary

Brief summary of what we implemented for Component 1 (for sharing with colleagues).

---

## What Component 1 Is

We train a **bi-encoder** so that math problems and curriculum standards live in a **shared embedding space**. The model learns to pull each problem close to its **correct standard(s)** and push it away from **hard negatives**—especially standards that are pedagogically nearby but wrong (siblings, conceptual neighbors, grade-adjacent). At inference we embed a new problem and retrieve top‑k standards by similarity (e.g. cosine or dot product). No hierarchy hints; we select from all 385 standards at once.

**Task:** Given a math word problem, output the set of Common Core standards it aligns with (multi-label over 385 standards).

---

## What We Implemented

| Piece | Location | Description |
|-------|----------|-------------|
| **Curriculum graph** | `mathfish/contrastive/curriculum_graph.py` | Builds from ATC: hierarchy (grade → domain → cluster → standard) + conceptual links. Exposes siblings, conceptual neighbors, grade-adjacent standards. |
| **Hard negative sampler** | `mathfish/contrastive/hard_negatives.py` | Samples negatives by ratio: 40% siblings, 30% conceptual, 20% grade-adjacent, 10% random. All from the graph/hierarchy; excludes gold standards for that problem. |
| **Contrastive dataset** | `mathfish/contrastive/contrastive_dataset.py` | One example per (problem, positive standard). Positive = ground-truth standard; negatives = sampler output (no gold in the negative set). |
| **Training** | `scripts/train_biencoder.py` | HuggingFace encoder (default: `sentence-transformers/all-MiniLM-L6-v2`) + separate 256-d projection heads for problem and standard, L2-normalized. InfoNCE loss, AdamW. Step-based training (e.g. 200 or 500 steps) with early stopping by loss. |
| **Inference** | `scripts/run_inference.py` | Embed problems and all standards with the trained model; output top‑k standard IDs per problem to a JSONL. |
| **Evaluation** | `scripts/evaluate_alignment.py` | Exact match, micro/macro F1, Recall@5/10/20, average graph distance, sibling confusion rate (uses same metrics as proposal). |

Data: MathFish problems (from HuggingFace `allenai/mathfish` or local JSONL) and ATC standards (from `allenai/achieve-the-core` or local paths). We use processed splits under `data/processed/` (e.g. `problems_train200.jsonl`, `problems_dev100.jsonl`).

---

## Positives and Negatives

- **Positive:** Always the **ground truth**—one of the problem’s gold standard IDs. If a problem has two aligned standards, we create two contrastive examples (same problem, different positive standard).
- **Negatives:** **Not** gold. Sampled via the **graph**: siblings (same cluster), conceptual neighbors (ATC coherence links), grade-adjacent standards, plus a small random fraction. This makes the model learn to separate “close but wrong” standards from the correct one.

### Exactly how we sample positives and negatives

**Positives**

1. Each training example is a (problem, one positive standard) pair.
2. We iterate over every problem and, for each problem, over every standard in that problem's **gold** `standard_ids` (from the MathFish labels).
3. For each such pair we create one dataset example: same problem text, that one standard as the positive. So a problem with 2 gold standards yields 2 examples; a problem with 1 gold standard yields 1 example. We only include standards that exist in our ATC `standard_id_to_text` map.
4. There is no randomness in choosing the positive—it is always one of the problem's ground-truth standard IDs.

**Negatives**

1. For each (problem, positive standard) example we call the hard-negative sampler with: the **current positive** standard ID, the **full set of gold standard IDs for that problem** (to exclude), and `num_negatives` (e.g. 4 or 8).
2. The sampler **excludes** all of that problem's gold standards from the negative set (so we never use a gold standard as a negative for the same problem).
3. It then fills `num_negatives` slots by type (with default ratios):
   - **Siblings (40%):** Standards that share the same **cluster** (parent in the ATC hierarchy) as the positive, minus the positive and any gold for this problem. From `CurriculumGraph.get_siblings(positive_id, exclude=gold_set)`.
   - **Conceptual (30%):** Standards that are **directly connected** to the positive in the ATC coherence graph (the "connections" in the standards data), minus the positive and gold. From `get_conceptual_neighbors(positive_id, exclude=gold_set)`.
   - **Grade-adjacent (20%):** Standards in the **same grade ±1** (e.g. grade 4 → grades 3 and 5), minus the positive and gold. From `get_grade_adjacent_standards(positive_id, exclude=gold_set)`.
   - **Random (10%):** Drawn from **all 385 standard IDs** (shuffled), skipping any already in the positive set or already chosen. No graph structure.
4. Within each category we shuffle and take without replacement until we have enough; if a category doesn't supply enough, the rest come from the next categories. Total negatives per example = `num_negatives` (no duplicates, and none in the problem's gold set).

So positives are fixed by the labels; negatives are sampled once per example (at `__getitem__` time) using the graph and hierarchy, with the above ratios.

---

## How to Run

```bash
# From mathfish_new/
cd mathfish_new

# 1. Train (e.g. 200 steps on 500 problems; use --from_hf to pull ATC from HuggingFace)
python scripts/train_biencoder.py \
  --problems_file data/processed/problems_train500.jsonl \
  --from_hf --max_steps 200 --output_dir outputs/biencoder

# 2. Inference (top-5 per problem)
python scripts/run_inference.py \
  --checkpoint_dir outputs/biencoder/checkpoint \
  --problems_file data/processed/problems_dev100.jsonl \
  --from_hf --output_file data/processed/predictions_dev100.jsonl --top_k 5

# 3. Evaluate
python scripts/evaluate_alignment.py \
  --predictions_file data/processed/predictions_dev100.jsonl \
  --gold_file data/processed/problems_dev100.jsonl \
  --standards_path data/cache/achieve-the-core/standards.jsonl \
  --domain_groups_path data/cache/achieve-the-core/domain_groups.json
```

One-liner (train → infer → eval) with defaults:

```bash
bash scripts/run_train_eval.sh
```

Use `TRAIN_N=2000`, `IMPROVED=1`, or `QUICK=--quick` as needed (see `docs/SPEED_UP_TRAINING.md`). **Note:** Training is step-based; with more data (e.g. 2000) you need more steps (e.g. 800+) to see each example multiple times, or results can be worse than with fewer data and the same step count.

---

## Current Status

- **Component 1:** Implemented and runnable (bi-encoder + curriculum-aware hard negatives + eval). Results so far are modest; tuning steps/epochs, data size, and hyperparameters is ongoing.
- **Component 2 (cross-encoder re-ranking):** Not implemented yet; `mathfish/reranker/` is a stub. The pipeline is currently bi-encoder only (retrieve top‑k, no re-ranker).

If you want more detail on any part (e.g. loss, encoder choice, or evaluation metrics), we can add a short subsection or point to the relevant files.
