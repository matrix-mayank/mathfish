# BGE-Large Training Guide

## 🚀 Changes for BGE-Large Training

### Training Notebook (`colab_train_biencoder_full.ipynb`)

**Cell 5 - Updated training command:**
```python
!python scripts/train_biencoder.py \
  --problems_file data/processed/problems_train_full.jsonl \
  --from_hf \
  --max_steps 15000 \
  --batch_size 32 \
  --num_negatives 8 \
  --temperature 0.03 \
  --early_stopping_patience 0 \
  --model_name "BAAI/bge-large-en-v1.5" \
  --fp16 \
  --output_dir outputs/biencoder_bge_large_15k
```

**Key changes:**
- ✅ `--model_name "BAAI/bge-large-en-v1.5"` (was: all-MiniLM-L6-v2)
- ✅ `--batch_size 32` (was: 64 - reduced for 335M param model)
- ✅ `--max_steps 15000` (was: 10000 - more steps for bigger model)
- ✅ `--output_dir outputs/biencoder_bge_large_15k`

**Expected:**
- Time: 60-90 minutes on H100/A100
- Output: `biencoder_bge_large_15k.zip` (~1.3GB)
- Performance: Recall@5 = 58-65% (vs 52% for MiniLM-L6)

---

## 📊 Model Comparison

| Model | Params | MTEB Score | Your R@5 | Training Time |
|-------|--------|------------|----------|---------------|
| MiniLM-L6 | 22M | 56.3 | 52% | 10-15 mins |
| **BGE-large** | **335M** | **70.5** | **58-65%?** | **60-90 mins** |

---

## 🔧 Inference Updates

When using the BGE-large model for inference:

**Upload the new checkpoint:**
- File: `biencoder_bge_large_15k.zip`
- Extract to: `outputs/biencoder_bge_large_15k/checkpoint/`

**Run inference:**
```python
!python scripts/run_inference.py \
  --checkpoint_dir outputs/biencoder_bge_large_15k/checkpoint \
  --problems_file data/processed/problems_dev_full.jsonl \
  --from_hf \
  --output_file outputs/biencoder_bge_top5.jsonl \
  --top_k 5
```

---

## ✅ What's Already Updated

I've updated `colab_train_biencoder_full.ipynb`:
- Cell 5: Changed to BGE-large with batch_size 32, 15k steps
- Cell 6: Updated checkpoint path
- Cell 7: Updated download filename
- Cell 8: Updated summary with BGE-large info

**Ready to use!** Just upload the notebook to Colab and run.

---

## 🎯 Next Steps

1. **Train BGE-large** (60-90 mins on H100)
2. **Download checkpoint** (1.3GB)
3. **Run inference** (15-20 mins)
4. **Compare results:**
   - MiniLM-L6: 52% R@5
   - BGE-large: ??? R@5 (hopefully 58-65%)

If BGE-large beats MiniLM-L6 by 5-10 points, you'll have SOTA retrieval! 🚀
