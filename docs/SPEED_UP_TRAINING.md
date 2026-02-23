# Speeding up training

Step 1 can feel stuck because (1) the first run downloads ATC + model from HuggingFace, and (2) training 200 steps on CPU is slow. Use these to speed things up.

## Quick run (fewer steps, shorter sequences)

```bash
python scripts/train_biencoder.py \
  --problems_file data/processed/problems_train500.jsonl \
  --from_hf --quick
```

`--quick` sets: **50 steps**, **max_len=128**, **batch_size=16**. Much faster than 200 steps with max_len=256.

## Explicit fast options (no --quick)

```bash
python scripts/train_biencoder.py \
  --problems_file data/processed/problems_train500.jsonl \
  --from_hf --max_steps 50 --max_len 128 --batch_size 16
```

- **--max_steps 50** — fewer steps (finish in a couple of minutes on CPU).
- **--max_len 128** — shorter sequences → faster forward/backward.
- **--batch_size 16** — more examples per step (fewer steps to cover the data).

## Use GPU + mixed precision (if you have CUDA)

```bash
python scripts/train_biencoder.py \
  --problems_file data/processed/problems_train500.jsonl \
  --from_hf --max_steps 200 --fp16
```

`--fp16` turns on mixed precision on GPU (often ~2× faster and less memory).

## Even smaller data

Train on 200 examples instead of 500:

```bash
python scripts/sample_train.py --n 200
python scripts/train_biencoder.py \
  --problems_file data/processed/problems_train200.jsonl \
  --from_hf --max_steps 50 --max_len 128
```

## One-command train + infer + eval

```bash
# Normal (200 steps)
bash scripts/run_train_eval.sh

# Quick (50 steps) — set QUICK=--quick
QUICK=--quick bash scripts/run_train_eval.sh

# Better accuracy — 500 steps, 8 negatives, lower temperature, save best checkpoint
IMPROVED=1 bash scripts/run_train_eval.sh

# Or fewer steps via env
TRAIN_STEPS=50 bash scripts/run_train_eval.sh
```

## Better accuracy (improved mode)

Use `--improved` (or `IMPROVED=1 bash scripts/run_train_eval.sh`) for higher quality:

- **500 steps** (or more) with **early-stopping patience 40**
- **8 hard negatives** per example (instead of 4)
- **Temperature 0.03** (sharper similarities)
- **Best checkpoint saved** — after training, the written checkpoint is the best by loss, not the last step

Expect higher recall@5 and F1 than the quick run; training takes longer (e.g. 15–30+ min on CPU).

## More training data

Training on 500 examples is the default. For better accuracy, use more data:

```bash
# Sample 2000 from MathFish train, then train with improved settings (saves to outputs/biencoder_2000)
TRAIN_N=2000 IMPROVED=1 bash scripts/run_train_eval.sh
```

The script will run `sample_train.py --n 2000` to create `data/processed/problems_train2000.jsonl` if that file doesn’t exist (or set `FORCE_SAMPLE=1` to re-sample). Checkpoint and predictions go to `outputs/biencoder_2000/` so they don’t overwrite the 500-example run.

## Why it can feel “stuck”

- **First run:** Downloads ATC (~few MB) and the MiniLM model (~90 MB). One-time.
- **CPU:** 200 steps can take 10–20+ minutes. Use `--max_steps 50` and `--max_len 128` to cut that.
- **Cursor/IDE:** Sometimes the integrated terminal kills long-running commands. Run in a normal terminal (e.g. Terminal.app or iTerm) so training isn’t aborted.
