# mathfish_new — Structure-Aware Contrastive Alignment

This folder contains the **custom project** build: contrastive bi-encoder + cross-encoder re-ranking for MathFish standard alignment, built on top of the original MathFish codebase (copied from `mathfish_original/`).

## Layout

- **`mathfish/`** — Main package
  - **`contrastive/`** — Curriculum graph, hard negative sampling, bi-encoder, contrastive dataset
  - **`reranker/`** — Cross-encoder re-ranker (stub)
  - **`alignment_eval/`** — Exact match, F1, Recall@k, graph distance, sibling confusion
  - **`tree_retriever/`**, **`evaluators/`**, **`preprocessors/`**, **`datasets/`** — From mathfish_original
- **`scripts/`**
  - **`build_curriculum_graph.py`** — Build ATC curriculum graph → `data/processed/curriculum_graph.json`
  - **`preprocess_mathfish.py`** — Load problems, clean, split train/val/test → `data/processed/problems_*.jsonl`
  - (To add: `train_biencoder.py`, `train_reranker.py`, `run_alignment_pipeline.py`, `evaluate_alignment.py`)
- **`data/processed/`** — Processed splits and curriculum graph
- **`configs/`** — Optional YAML configs

## Data

- **Standards / domain groups:** Download from HuggingFace [allenai/achieve-the-core](https://huggingface.co/datasets/allenai/achieve-the-core) (or use local paths). Pass `--standards_path` and `--domain_groups_path` to scripts.
- **Problems:** HuggingFace [allenai/mathfish](https://huggingface.co/datasets/allenai/mathfish) or a local jsonl with keys `id`, `problem_activity`/`content`/`text`, `standards`, `elements`.

## Quick start

```bash
# From repo root (mathfish_new/)
pip install -e .

# Build curriculum graph (need standards + domain_groups)
python scripts/build_curriculum_graph.py \
  --standards_path /path/to/standards.jsonl \
  --domain_groups_path /path/to/domain_groups.json

# Preprocess problems and split
python scripts/preprocess_mathfish.py \
  --hf_dataset allenai/mathfish \
  --standards_path /path/to/standards.jsonl \
  --output_dir data/processed
```

## Milestones

1. **Data & graph** — Done: `build_curriculum_graph.py`, `preprocess_mathfish.py`, `CurriculumGraph`, `HardNegativeSampler`
2. **Contrastive data** — Done: `ContrastiveDataset`
3. **Bi-encoder** — Stub: `BiEncoder`; next: training script + InfoNCE loss
4. **Cross-encoder** — Pending
5. **Evaluation** — Done: `alignment_eval` metrics and `AlignmentEvaluator`
6. **Baselines & ablations** — Pending
