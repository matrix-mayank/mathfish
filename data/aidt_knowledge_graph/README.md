# Knowledge Graph Dataset

This directory contains the Knowledge Graph dataset downloaded from the public S3 bucket.

## Files

### Entity Files (Nodes)

1. **StandardsFramework.csv/json** (114KB / 158KB)
   - Educational standards frameworks
   - Contains framework-level metadata

2. **StandardsFrameworkItem.csv/json** (134MB / 190MB)
   - Individual standards and learning objectives within frameworks
   - Contains standards, clusters, domains, etc.

3. **LearningComponent.csv/json** (979KB / 1.3MB)
   - Granular, precise representations of individual skills or concepts
   - More detailed breakdown than standards

### Relationship Files (Edges)

4. **Relationships.csv/json** (151MB / 210MB)
   - Connections and associations between all entity/node types
   - Format: source UUID → relationship type → target UUID

## Data Format

- **CSV files**: UTF-8 encoded with comma delimiters and quoted fields. Includes header rows.
- **JSON files**: Newline delimited JSON (NDJSON) format with UTF-8 encoding.

## Usage

These files can be used to:
- Build graph structures for GNN applications
- Compare with MathFish dataset structure
- Support standards alignment and tagging tasks
- Enable cross-state standards comparison

## Download Date

Downloaded: February 5, 2025
Source: https://aidt-knowledge-graph-datasets-public-prod.s3.us-west-2.amazonaws.com/knowledge-graph/v1.0.0/

## Related Files

See `MATHFISH_VS_KNOWLEDGE_GRAPH.md` in the project root for a detailed comparison between MathFish and Knowledge Graph datasets.

