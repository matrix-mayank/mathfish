# Data Directory Structure

This directory contains datasets used in the MathFish project, organized into separate folders for clarity.

## Directory Structure

```
data/
├── mathfish/                    # MathFish dataset files
│   ├── few_shot/                # Few-shot examples for prompts
│   ├── generated_problems_as_input_data.jsonl
│   ├── generated_problems_teacher_annotations.csv
│   └── gsm8k_test.jsonl
│
└── aidt_knowledge_graph/        # AIDT Knowledge Graph dataset (for comparison)
    ├── StandardsFramework.csv/json
    ├── StandardsFrameworkItem.csv/json
    ├── LearningComponent.csv/json
    └── Relationships.csv/json
```

## MathFish Data

The `mathfish/` folder contains:
- **Few-shot examples**: Used for prompt engineering in verification and tagging tasks
- **Generated problems**: LLM-generated math problems with teacher annotations
- **GSM8K test set**: Reformatted for MathFish evaluation

**Note**: The main MathFish standards data (`standards.jsonl` and `domain_groups.json`) should be downloaded from HuggingFace:
- [achieve-the-core](https://huggingface.co/datasets/allenai/achieve-the-core)
- [mathfish](https://huggingface.co/datasets/allenai/mathfish)

These are passed as command-line arguments (`--standards_path`, `--domain_groups_path`) to scripts, so they can be stored anywhere.

## AIDT Knowledge Graph Data

The `aidt_knowledge_graph/` folder contains the Knowledge Graph dataset from AIDT (AI Data Tools) for comparison and research purposes. This is a separate dataset with a similar graph structure to MathFish but covering multiple subjects and states.

See `MATHFISH_VS_KNOWLEDGE_GRAPH.md` in the project root for a detailed comparison.

## Impact on MathFish Scripts

**No impact**: MathFish scripts use command-line arguments to specify data paths, so organizing data into separate folders does not affect functionality. The Knowledge Graph data is completely separate and will not interfere with MathFish operations.


