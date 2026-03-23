# Knowledge Base Subgraph Extractor

This script extracts subgraphs from a knowledge base (e.g., Wikidata) by traversing multi-hop neighborhoods around seed entities. It produces (subject, predicate, object) triples that form the subgraph surrounding each entity.

## Usage

```bash
python knowledge_base_triple_extractor.py --multiple_samples --num_samples 10 --max_hops 6 --parallel --num_threads 2 --controlled_extraction --num_neighbors_per_hop 4 --source wikidata --type_qid Q8502 --resume_generation
```

### Argument Reference

| Argument | Description |
|----------|-------------|
| `--multiple_samples` | Enable batch mode: extract subgraphs for multiple entities instead of a single entity. |
| `--num_samples` | Number of entity subgraphs to extract (e.g., `10`). |
| `--max_hops` | Maximum neighborhood depth. Entities up to 6 hops away from the seed are included in the subgraph. |
| `--parallel` | Run extraction in parallel across multiple threads. |
| `--num_threads` | Number of worker threads for parallel extraction (e.g., `2`). |
| `--controlled_extraction` | Use controlled expansion: limit how many neighbors are added at each hop to keep subgraphs manageable. |
| `--num_neighbors_per_hop` | Maximum neighbors to expand per hop in controlled mode (e.g., `4`). |
| `--source` | Knowledge base to query (e.g., `wikidata`). |
| `--type_qid` | Wikidata QID for entity type filter. Only entities of this type are used as seeds (e.g., `Q8502` for mountains). You can use any valid Wikidata type QID. |
| `--resume_generation` | Skip entities that already have output files and only process new ones. Useful for resuming interrupted runs. |

---

# Text Description Generator for Extracted Knowledge Graphs

This script generates natural language text descriptions for extracted knowledge graph subgraphs. Given triples (subject, predicate, object), it uses an LLM to produce readable textual summaries suitable for training and evaluation.

## Usage

```bash
python generate_text_from_kg.py --source wikidata --model deepseek-ai/DeepSeek-V3 --llm_provider deepseek --postfix "_triples.txt"
```

```bash
python generate_text_from_kg.py --source wikidata --model gpt-3.5-turbo --llm_provider chatgpt
```

### Argument Reference

| Argument | Description |
|----------|-------------|
| `--source` | Data source (e.g., `wikidata`). |
| `--model` | LLM model to use (e.g., `deepseek-ai/DeepSeek-V3`, `gpt-3.5-turbo`, `gpt-4o`). |
| `--llm_provider` | LLM API provider: `chatgpt` or `deepseek`. |
| `--postfix` | File postfix for triple files to process (e.g., `_triples.txt`, `_triples_pruned.txt`). |
| `--num_threads` | Number of threads for parallel generation (default: 1). |
| `--skip_existing` | Skip files that already have a corresponding `_text.txt` output; useful for resuming interrupted runs. |

---

# partition_files.py
```bash
python partition_files.py --source wikidata --num_partitions 3
```