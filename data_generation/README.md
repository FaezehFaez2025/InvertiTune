# Knowledge Base Subgraph Extractor

This script extracts local subgraphs from **Wikidata** by traversing multi-hop neighborhoods around seed entities. It produces (subject, predicate, object) triples that form the subgraph surrounding each entity.

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

# calculate_pruned_triple_statistics.py

This script calculates statistics for triples stored in `_pruned.txt` files within the `data` directory. It provides the number of files, average number of triples per file, maximum/minimum number of triples per file, standard deviation, and a distribution of the number of triples across files.

**Command to run:**
```bash
python calculate_pruned_triple_statistics.py --source wikidata
```

**Additional commands:**
```bash
python calculate_pruned_triple_statistics.py --source wikidata --postfix _triples.txt
```

```bash
python calculate_pruned_triple_statistics.py --source wikidata --postfix _triples.txt --path ../data/test/
```

```bash
python calculate_pruned_triple_statistics.py --source wikidata --postfix _triples.txt --path ../data/train/
```

# generate_text_from_kg.py
```bash
python generate_text_from_kg.py --source wikidata --model deepseek-ai/DeepSeek-V3 --llm_provider deepseek --postfix "_triples.txt"
```

```bash
python generate_text_from_kg.py --source wikidata --model gpt-3.5-turbo --llm_provider chatgpt
```
## Parallel Text Generation
```bash
python generate_text_from_kg.py --source wikidata --model deepseek-ai/DeepSeek-V3 --llm_provider deepseek --postfix "_triples.txt" --num_threads 10
```
## Skip Existing Flag
```bash
python generate_text_from_kg.py --source wikidata --model deepseek-ai/DeepSeek-V3 --llm_provider deepseek --postfix "_triples.txt" --num_threads 10 --skip_existing
```

# partition_files.py
```bash
python partition_files.py --source wikidata --num_partitions 3
```