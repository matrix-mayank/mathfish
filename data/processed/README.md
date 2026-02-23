# Processed data (generated)

- **problems_train.jsonl**, **problems_val.jsonl**, **problems_test.jsonl** — Produced by `scripts/preprocess_mathfish.py`. Each line: `{"id": str, "text": str, "standard_ids": [str]}`.
- **curriculum_graph.json** — Produced by `scripts/build_curriculum_graph.py` from ATC standards and domain groups.

Run the scripts from `mathfish_new/` with appropriate `--standards_path` and `--domain_groups_path` (e.g. from HuggingFace `allenai/achieve-the-core`).
