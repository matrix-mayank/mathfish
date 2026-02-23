# Expand dataset, rerun training, and evaluate

## 1. Expand the dataset (use full train/val/test from HF)

Use the paper’s splits so evaluation is comparable:

```bash
cd mathfish_new
python scripts/preprocess_mathfish.py --use_hf_splits
```

This loads **train**, **dev**, **test** from HuggingFace and writes:

- `data/processed/problems_train.jsonl`
- `data/processed/problems_val.jsonl`
- `data/processed/problems_test.jsonl`

Optional: standardize standard IDs (use cached ATC after `build_curriculum_graph.py --from_hf`):

```bash
python scripts/preprocess_mathfish.py --use_hf_splits \
  --standards_path data/cache/achieve-the-core/standards.jsonl
```

## 2. Rerun training (full train set, more steps)

Train on the full train split with more steps:

```bash
python scripts/train_biencoder.py \
  --problems_file data/processed/problems_train.jsonl \
  --from_hf \
  --max_steps 2000 \
  --batch_size 16 \
  --output_dir outputs/biencoder_full
```

Adjust `--max_steps` and `--batch_size` as needed (e.g. 500–2000 steps for a quick run, more for a proper run).

## 3. Run inference (predict on test or dev)

Produce predicted standard IDs using the trained bi-encoder:

```bash
python scripts/run_inference.py \
  --checkpoint_dir outputs/biencoder_full/checkpoint \
  --problems_file data/processed/problems_test.jsonl \
  --from_hf \
  --output_file data/processed/predictions_test.jsonl \
  --top_k 5
```

Use `--top_k` to match how many standards you want per problem (e.g. 5 or 10). For exact-match evaluation, you can try `top_k` equal to the typical number of gold standards per problem.

## 4. Evaluate

Run the alignment evaluator on the predictions:

```bash
python scripts/evaluate_alignment.py \
  --predictions_file data/processed/predictions_test.jsonl \
  --gold_file data/processed/problems_test.jsonl \
  --standards_path data/cache/achieve-the-core/standards.jsonl \
  --domain_groups_path data/cache/achieve-the-core/domain_groups.json \
  --output_file results/eval_test.json
```

If you used `--from_hf` for preprocessing, the cache path is `data/cache/achieve-the-core/`. If you built the graph with `build_curriculum_graph.py --from_hf`, the same cache is used.

You’ll get **exact match**, **micro/macro F1**, **Recall@k**, and (if graph paths are given) **avg graph distance** and **sibling confusion rate**.

---

## Quick recap

| Step | Command / action |
|------|-------------------|
| 1. Expand data | `preprocess_mathfish.py --use_hf_splits` |
| 2. Train | `train_biencoder.py --problems_file .../problems_train.jsonl --max_steps 2000` |
| 3. Infer | `run_inference.py --checkpoint_dir ... --problems_file .../problems_test.jsonl` |
| 4. Eval | `evaluate_alignment.py --predictions_file ... --gold_file .../problems_test.jsonl` |

After that you can compare runs (e.g. 100-sample vs full train) or add the cross-encoder re-ranker and evaluate again.
