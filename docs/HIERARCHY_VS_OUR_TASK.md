# Domain → Cluster → Standard vs Our Task

## What your friend meant: the **original MathFish tagging** (three steps)

In the **original paper**, one of the tagging setups is **hierarchical traversal**:

1. **Domain** — Model is given options like "Number & Operations", "Geometry", … and picks one or more.
2. **Cluster** — For each chosen domain, model is given cluster options (e.g. "Understand place value") and picks.
3. **Standard** — For each chosen cluster, model is given standard options and picks the final standard(s).

So the model **does** follow **domain → cluster → standard** when the task is set up that way (with options at each level). That’s the **assisted** setting: the hierarchy is revealed step by step.

## What **we** are doing: **flat** prediction (no three steps)

Our **custom project** (from the proposal) uses a **different**, harder setting:

- **Input:** One math problem (text).
- **Output:** The **set of Common Core standard IDs** (from all **385** standards) that the problem aligns with.
- **No** domain → cluster → standard steps: the model is **not** asked to choose domain first, then cluster, then standard. It must produce standard IDs **directly** (e.g. via retrieval over 385 standards, then optional re-ranking).

So we are **not** following the three-step traversal. We use the **curriculum structure** (domain, cluster, siblings, conceptual links) **inside** the model:

- For **training**: hard negatives are chosen using that structure (siblings, conceptual neighbors, grade-adjacent).
- For **evaluation**: we still compare **predicted set of standards** vs **gold set** (exact match, F1, Recall@k, graph distance).

The hierarchy is used to make the model better at distinguishing nearby standards; we don’t require the model to output domain → cluster → standard explicitly.

## Summary

| | Original MathFish (assisted tagging) | Our project |
|---|--------------------------------------|-------------|
| **Task** | Choose at each level: domain → cluster → standard | Predict **set of standard IDs** from 385 at once |
| **Three steps?** | Yes (domain → cluster → standard) | No (flat prediction of standards) |
| **Use of hierarchy** | Shown to the model as the interface | Used for hard negatives and evaluation metrics |

So: **domain → cluster → standard** is the **original** tagging interface; **we** do flat standard prediction and use the hierarchy only for training and evaluation.
