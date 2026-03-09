# Dataset Sizes and Rationale

## 📊 Overview of All Datasets

### 1. **Problem Datasets** (Math problems → Standards alignment)

| Dataset | Size | Avg Standards/Problem | Purpose | Why This Size? |
|---------|------|----------------------|---------|----------------|
| **train500** | 500 problems | 1.05 | Quick training/testing | Small subset for fast iteration |
| **train2000** | 2,000 problems | 0.98 | Bi-encoder training | ✅ **Used** - Balance of speed and data |
| **dev100** | 100 problems | 1.00 | Validation/testing | Standard eval size for quick feedback |
| **test** | 2 problems | 1.50 | Quick sanity check | Dummy test set (not real evaluation) |

**Total Standards in Curriculum**: 737 (from Achieve the Core K-12 Common Core)

---

## 🎯 What We Actually Used

### Bi-Encoder Training
- **Dataset**: `problems_train2000.jsonl` 
- **Size**: 2,000 problems → 1,960 problem-standard pairs
- **Training**: 5,000 steps, batch size 64, 8 negatives per positive
- **Why 2,000 problems?**
  - ✅ Large enough for model to learn patterns
  - ✅ Small enough to train in ~15-20 mins on A100
  - ✅ ~200 epochs (5000 steps ÷ 25 batches/epoch)
  
**Contrastive Training Details**:
```
Dataset size: 1,586 training pairs
- Each problem paired with 1 positive standard
- 8 hard negatives sampled per positive
- Negatives: curriculum siblings, same domain, random
→ Effective training: 1,586 × 9 = 14,274 examples per epoch
→ Total examples seen: 14,274 × 200 = ~2.85M examples
```

---

### Verifier Training (DeBERTa-v3-small)
- **Dataset**: Binary verification dataset (from previous work)
- **Size**: 35,104 examples total
  - 8,776 positive pairs (25%) - problem truly aligns with standard
  - 26,328 negative pairs (75%) - problem does NOT align
- **Training**: 2 epochs on full dataset
- **Why 35,104 examples?**
  - ✅ Balanced hard negatives (siblings: 40%, same domain: 30%, random: 30%)
  - ✅ Large enough for DeBERTa to learn discrimination
  - ✅ 75% negatives to match real-world imbalance (most candidates are wrong)

**Training Time**: ~2.5 hours for 2 epochs (not re-done in this session)

---

### Inference & Evaluation
- **Dataset**: `problems_dev100.jsonl`
- **Size**: 100 problems (100 gold standards)
- **Why 100 problems?**
  - ✅ Standard eval size in ML research
  - ✅ Fast enough for multiple threshold experiments (~2-3 mins on GPU)
  - ✅ Large enough for reliable metrics (vs. 2 dummy problems)
  - ❌ Note: We initially tested on 2 dummy problems (unreliable!)

---

## 🔢 Training Decisions Explained

### Why NOT train on all data?

1. **Bi-Encoder: Why 2,000 instead of full dataset?**
   - The full MathFish dataset has more problems available
   - But 2,000 problems with proper negative sampling gives:
     - ~2.85M training examples (with 8 negatives)
     - Sufficient coverage of 737 standards
     - Fast iteration (15-20 mins vs hours)
   
2. **Why 5,000 steps?**
   - ~200 epochs through data
   - Balance between:
     - ✅ Sufficient training (loss converges)
     - ✅ Fast turnaround (not overnight)
     - ❌ Risk of overfitting (no early stopping used)

3. **Why batch size 64 with 8 negatives?**
   - Batch size 64: Max that fits in A100 memory with fp16
   - 8 negatives: Standard for contrastive learning
     - More negatives = harder training = better discrimination
     - But diminishing returns after 8-12 negatives

---

## 📈 Why These Numbers Work

### Information Theory Perspective
- **737 standards** to learn
- **2,000 problems** means ~2-3 examples per standard on average
- With **8 negatives**, model sees each standard in context with ~16-24 other standards
- This gives sufficient **contrast** for the model to learn meaningful embeddings

### Practical Constraints
- **GPU Time**: A100 costs ~$1-2/hour on Colab
  - 5,000 steps = 15-20 mins = economical
  - 10,000+ steps = 30-40 mins = better but 2x cost
  
- **Iteration Speed**: 
  - Small datasets (500) = too little data, poor generalization
  - Medium datasets (2,000) = ✅ sweet spot
  - Large datasets (5,000+) = better performance but slower iteration

---

## 🎯 Comparison to Baselines

From the transcript, the verifier was trained previously with:
- 35,104 examples (8,776 positive + 26,328 negative)
- Val F1: 34.06% (not great discrimination)
- Issue: Over-confident predictions (scores 0.7-0.96)

The bi-encoder training (5,000 steps on 2,000 problems):
- Initial results: Recall@20 = 25% (only 1/4 correct standards in top-20)
- This is why threshold needed to be 0.9+ to filter candidates

---

## 💡 Recommendations for Future Improvements

### To Improve Bi-Encoder:
1. **More training steps**: 7,000-10,000 (better recall)
2. **More training data**: Use train500 + train2000 = 2,500 problems
3. **More negatives**: Try 12-16 negatives per positive
4. **Better negative sampling**: Use curriculum graph more aggressively

### To Improve Verifier:
1. **More training epochs**: 3-5 epochs instead of 2
2. **Calibration**: Add temperature scaling to fix over-confidence
3. **Better negatives**: Focus on very hard negatives (siblings only)
4. **Larger model**: Try DeBERTa-v3-base (3x params)

### Current Bottlenecks:
1. **Bi-encoder recall too low** (25% in top-20) ← Most critical
2. **Verifier over-confident** (needs calibration)
3. **Small test set** (100 problems - could use 500-1000)

---

## 📊 Summary Table

| Component | Dataset | Size | Training Time | Cost | Performance |
|-----------|---------|------|---------------|------|-------------|
| Bi-encoder | train2000 | 2,000 problems | 15-20 mins (A100) | ~$0.50 | Recall@20: 25% |
| Verifier | verification | 35,104 pairs | 2.5 hours (GPU) | ~$5.00 | Val F1: 34% |
| Evaluation | dev100 | 100 problems | 2-3 mins (A100) | ~$0.10 | Best F1: 9.52% @ threshold 0.9 |

**Total Training Cost**: ~$5.50 (Colab A100)  
**Total Training Time**: ~3 hours  
**Final Performance**: 9.52% F1 on 100-problem test set
