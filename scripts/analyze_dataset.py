"""
Check how many problems in the full MathFish dataset have standards labels.
Downloads from HuggingFace and reports statistics.
"""
import json
from huggingface_hub import hf_hub_download

def analyze_dataset(split_name):
    """Download and analyze a split of MathFish dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {split_name.upper()} split")
    print(f"{'='*60}")
    
    # Download the file
    path = hf_hub_download(
        repo_id="allenai/mathfish",
        filename=f"{split_name}.jsonl",
        repo_type="dataset"
    )
    
    # Parse line by line
    total_problems = 0
    problems_with_standards = 0
    total_standard_labels = 0
    problems_by_standard_count = {}
    
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                total_problems += 1
                
                # Extract standards
                standards = row.get("standards") or []
                # Standards are in format: [(relation, standard_id), ...]
                standard_ids = []
                for s in standards:
                    if isinstance(s, (list, tuple)) and len(s) > 1:
                        relation, std_id = s[0], s[1]
                        # Only count "Addressing" or "Alignment" relations
                        if relation in ["Addressing", "Alignment"]:
                            standard_ids.append(std_id)
                    elif isinstance(s, str):
                        standard_ids.append(s)
                
                num_standards = len(standard_ids)
                if num_standards > 0:
                    problems_with_standards += 1
                    total_standard_labels += num_standards
                    problems_by_standard_count[num_standards] = problems_by_standard_count.get(num_standards, 0) + 1
                    
            except json.JSONDecodeError:
                continue
    
    # Report statistics
    print(f"\n📊 Statistics:")
    print(f"  Total problems: {total_problems:,}")
    print(f"  Problems with standards: {problems_with_standards:,} ({100*problems_with_standards/total_problems:.1f}%)")
    print(f"  Problems without standards: {total_problems - problems_with_standards:,} ({100*(total_problems - problems_with_standards)/total_problems:.1f}%)")
    print(f"  Total standard labels: {total_standard_labels:,}")
    print(f"  Average standards per labeled problem: {total_standard_labels/problems_with_standards:.2f}")
    
    print(f"\n📈 Distribution of standards per problem:")
    for count in sorted(problems_by_standard_count.keys()):
        num_probs = problems_by_standard_count[count]
        print(f"  {count} standard(s): {num_probs:,} problems ({100*num_probs/problems_with_standards:.1f}%)")
    
    return {
        'split': split_name,
        'total': total_problems,
        'with_standards': problems_with_standards,
        'total_labels': total_standard_labels,
        'avg_per_problem': total_standard_labels/problems_with_standards if problems_with_standards > 0 else 0
    }

if __name__ == "__main__":
    print("🔍 Checking MathFish dataset for standards labels...")
    print("This will download data from HuggingFace...")
    
    results = []
    for split in ['train', 'dev', 'test']:
        try:
            result = analyze_dataset(split)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error analyzing {split}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Split':<10} {'Total':<10} {'Labeled':<10} {'% Labeled':<12} {'Avg Standards'}")
    print("-" * 60)
    for r in results:
        pct = 100 * r['with_standards'] / r['total'] if r['total'] > 0 else 0
        print(f"{r['split']:<10} {r['total']:<10,} {r['with_standards']:<10,} {pct:<12.1f} {r['avg_per_problem']:.2f}")
    
    print(f"\n✅ Analysis complete!")
    print(f"\n💡 Recommendation:")
    total_labeled = sum(r['with_standards'] for r in results)
    print(f"   Use {total_labeled:,} labeled problems for training/evaluation")
    print(f"   (Filter out problems without standards labels)")
