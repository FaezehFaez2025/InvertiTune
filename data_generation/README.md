# knowledge_base_triple_extractor.py

This Python script extracts triples (subject, predicate, object) from **Wikidata** or **YAGO** knowledge bases. It retrieves n-hop neighbors of a given entity and includes an optional GUI to select 
and save some of the triples.

## Usage

### Command-Line Arguments
- `--entity`: The entity ID to query (e.g., `Q12345` for Wikidata or `ACF_Fiorentina` for YAGO).
- `--hops`: The number of hops (default: 1).
- `--save`: Enable the GUI to select and save some of the triples.
- `--source`: The knowledge base to query (`wikidata` or `yago`). Default is `wikidata`.

### Example Commands

#### Query Wikidata
```bash
python knowledge_base_triple_extractor.py --entity Q12345 --hops 2 --save
```
#### Query YAGO:
```bash
python knowledge_base_triple_extractor.py --entity ACF_Fiorentina --hops 2 --source yago --save
```
#### TBC
```bash
python knowledge_base_triple_extractor.py --multiple_samples --num_samples 5000 --max_hops 2 --source wikidata --ratio 0.2
```
#### Parallel Extraction (TBC)
```bash
python knowledge_base_triple_extractor.py --multiple_samples --num_samples 10 --max_hops 2 --source wikidata --ratio 0.3 --parallel --num_threads 4
```
#### Controlled Extraction (TBC)
```bash
python knowledge_base_triple_extractor.py --multiple_samples --num_samples 10 --max_hops 4 --parallel --num_threads 2 --controlled_extraction --num_neighbors_per_hop 3 --source wikidata
```
#### Controlled Extraction with Entity Type Constraints (TBC)
```bash
python knowledge_base_triple_extractor.py --multiple_samples --num_samples 2 --max_hops 4 --parallel --num_threads 2 --controlled_extraction --num_neighbors_per_hop 5 --source wikidata --type_qid Q5
```
#### Resume Generation
```bash
python knowledge_base_triple_extractor.py --multiple_samples --num_samples 200 --max_hops 4 --parallel --num_threads 5 --controlled_extraction --num_neighbors_per_hop 6 --source wikidata --type_qid Q5 --resume_generation
``` 
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
# rule_based_triple_filtering.py
```bash
python rule_based_triple_filtering.py --source wikidata
```
# entity_triple_viewer.py

This script retrieves all relationships where the specified entity is the subject. It queries Wikidata to extract and display all triples associated with a given entity ID.

## Usage

```bash
python entity_triple_viewer.py Q6581097
```

Replace `Q6581097` with any Wikidata entity ID you want to explore.

# entity_expansion_evaluator.py

Evaluates if a Wikidata entity is worth expanding based on triple informativeness. First applies rule-based filtering, then uses LLM prompting on the remaining triples to determine if all are non-informative or if any informative triples exist.

```bash
python entity_expansion_evaluator.py Q5 --model gpt-4o --batch_size 1
```
