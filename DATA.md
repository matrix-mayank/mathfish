# Data: Do we have it? Do you need an API key?

## Short answers

- **We do not have the full MathFish dataset in the repo.** The ~9.9k problems and ATC standards live on HuggingFace.
- **We use the same data as the original MathFish paper:** `allenai/mathfish` and `allenai/achieve-the-core`.
- **No API key or login needed.** Both are public datasets.

---

## Yes: you can use HuggingFace directly (no local download step)

Scripts can pull from HuggingFace without running a separate "download" step. The `datasets` / `huggingface_hub` libraries cache automatically.

### Preprocess problems (directly from HF)

Uses the paper's train/dev/test splits from HF; nothing is written under `data/raw/` for problems:

```bash
cd mathfish_new
pip install datasets

# Use paper splits: train → train, dev → val, test → test (no random split)
python scripts/preprocess_mathfish.py --use_hf_splits
```

Output: `data/processed/problems_train.jsonl`, `problems_val.jsonl`, `problems_test.jsonl`.  
Optional: add `--standards_path <path>` to standardize standard IDs (e.g. path to cached ATC standards; see below).

### Build curriculum graph (directly from HF)

First run downloads ATC into a small cache dir; later runs reuse it:

```bash
pip install huggingface_hub
python scripts/build_curriculum_graph.py --from_hf
```

That uses `allenai/achieve-the-core` and caches to `data/cache/achieve-the-core/` by default. No need to pass `--standards_path` or `--domain_groups_path`.

### Optional: standardize IDs when preprocessing

If you ran `build_curriculum_graph.py --from_hf`, you can point preprocess at the cached standards:

```bash
python scripts/preprocess_mathfish.py --use_hf_splits \
  --standards_path data/cache/achieve-the-core/standards.jsonl
```

---

## Alternative: download to local files first

If you prefer to have everything under `data/raw/`:

```bash
python scripts/download_mathfish_data.py
```

Then use `--input_file`, `--standards_path`, and `--domain_groups_path` as before (see README or script help).

---

## When would you need a HuggingFace token?

Only for **gated models** or **private repos**. For `allenai/mathfish` and `allenai/achieve-the-core`, no token is required.
