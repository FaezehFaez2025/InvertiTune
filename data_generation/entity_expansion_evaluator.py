import os
import argparse
import logging
import subprocess
import sys
from openai import OpenAI
from dotenv import load_dotenv
from entity_triple_viewer import query_entity_triples, get_entity_name
from rule_based_triple_filtering import filter_triple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_entity_triples_for_evaluation(entity_id):
    """
    Get all triples for an entity using the entity_triple_viewer functionality.
    Apply rule-based filtering to remove obviously non-informative triples.
    Returns the entity name and list of filtered triples.
    """
    try:
        entity_name = get_entity_name(entity_id)
        if not entity_name:
            logging.error(f"Could not find entity name for ID '{entity_id}'")
            return None, []
        
        triples = query_entity_triples(entity_id)
        original_count = len(triples)
        
        # Apply rule-based filtering
        logging.info(f"Applying rule-based filtering to {len(triples)} triples...")
        filtered_triples = []
        for triple in triples:
            # Convert triple to string format for the existing filter_triple function
            triple_str = f'["{triple[0]}", "{triple[1]}", "{triple[2]}"]'
            if filter_triple(triple_str):
                filtered_triples.append(triple)
        
        print(f"Rule-based filtering: {original_count} -> {len(filtered_triples)} triples")
        logging.info(f"Rule-based filtering: {len(triples)} -> {len(filtered_triples)} triples")
        return entity_name, filtered_triples
    except Exception as e:
        logging.error(f"Error retrieving triples for entity {entity_id}: {e}")
        return None, []

def evaluate_triple_batch(triples_batch, entity_name, entity_id, model, llm_provider, client):
    """
    Evaluate a batch of triples for informativeness.
    Returns True if ALL triples in the batch are non-informative.
    """
    # Format triples for the prompt
    triples_text = "\n".join([f'["{t[0]}", "{t[1]}", "{t[2]}"]' for t in triples_batch])

    prompt = f"""
You are an expert in knowledge graph analysis.  
Decide if a batch of triples about an entity contains only NON-INFORMATIVE knowledge.

### Definition
- **NON-INFORMATIVE**: trivial, obvious, generic, or vague facts that do not add meaningful knowledge.  
  This includes:
  - **Common sense or obvious traits** (e.g., humans are mortal, fire is hot)  
  - **Basic opposites or simple taxonomic facts** (e.g., male opposite of female, male different from man/masculinity)  
  - **Overly broad or vague relations** that apply to almost any entity  
    (e.g., human has effect artificial object, human interacts with environment)  

- **INFORMATIVE**: specific, distinctive, or non-obvious facts that provide concrete knowledge about the entity  
  (e.g., birthplace, achievements, historical events, numerical values).

### Examples
NON-INFORMATIVE:
["male", "opposite of", "female"]  
["human", "has characteristic", "mortality"]  
["minus sign", "opposite of", "plus sign"]  
["human", "has effect", "artificial object"]  
["human", "physically interacts with", "natural environment"]  

INFORMATIVE:
["Albert Einstein", "born in", "Ulm"]  
["Albert Einstein", "developed", "theory of relativity"]  
["Paris", "capital of", "France"]  
["Paris", "population", "2,161,000"]  

### Task
Entity: {entity_name} ({entity_id})  
Triples:  
{triples_text}

### Question
Are **all** of these triples NON-INFORMATIVE?

### Output (STRICT)
YES  (all non-informative)  
NO   (at least one informative)
"""

    try:
        # Call the API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in knowledge graph analysis who evaluates the informativeness of triple batches."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for consistent evaluation
        )
        
        # Extract the response
        evaluation_response = response.choices[0].message.content.strip()
        
        # Parse the response
        all_non_informative = evaluation_response.upper().startswith("YES")
        
        return all_non_informative, evaluation_response
        
    except Exception as e:
        logging.error(f"An error occurred while calling the {llm_provider.upper()} API: {e}")
        return None, None

def evaluate_entity_expansion_worthiness(entity_id, entity_name, triples, model, llm_provider, batch_size=10):
    """
    Use LLM to evaluate if an entity is worth expanding based on its triples.
    Processes triples in batches for better accuracy.
    Returns True if entity should NOT be expanded (all triples are non-informative).
    Also returns the informative triples found.
    """
    # Define the relative path to the .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
    load_dotenv(dotenv_path)

    # Get the API key from the .env file
    if llm_provider == "deepseek":
        api_key = os.getenv("NEBIUS_API_KEY")
        base_url = "https://api.studio.nebius.com/v1/"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None

    if not api_key:
        raise ValueError(f"Please set your API key for {llm_provider} in the .env file.")

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Split triples into batches
    batches = [triples[i:i + batch_size] for i in range(0, len(triples), batch_size)]
    
    logging.info(f"Processing {len(triples)} triples in {len(batches)} batches of up to {batch_size} triples each")
    
    batch_results = []
    all_responses = []
    informative_triples = []
    
    # Evaluate each batch
    for i, batch in enumerate(batches):
        logging.info(f"Evaluating batch {i+1}/{len(batches)} ({len(batch)} triples)")
        
        all_non_informative, response = evaluate_triple_batch(
            batch, entity_name, entity_id, model, llm_provider, client
        )
        
        if all_non_informative is None:
            logging.error(f"Failed to evaluate batch {i+1}")
            return None, None, []
        
        batch_results.append(all_non_informative)
        all_responses.append(f"Batch {i+1}: {response}")
        
        # If this batch contains informative triples, add them to the collection
        if not all_non_informative:
            logging.info(f"Found informative triples in batch {i+1}")
            # Add all triples from this batch as potentially informative
            informative_triples.extend(batch)
    
    # Determine if entity should be expanded based on whether any batch contained informative triples
    has_informative = len(informative_triples) > 0
    
    if has_informative:
        logging.info(f"Found informative triples in {len([r for r in batch_results if not r])} batch(es) - entity should be expanded")
        return False, "\n".join(all_responses), informative_triples
    else:
        logging.info("All batches contain only non-informative triples - entity should not be expanded")
        return True, "\n".join(all_responses), []

def main():
    parser = argparse.ArgumentParser(description="Evaluate if a Wikidata entity is worth expanding based on the informativeness of its triples.")
    parser.add_argument("entity", type=str, help="Wikidata entity ID to evaluate (e.g., Q6581097)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="The model to use. Available options: gpt-4o, gpt-4o-mini, gpt-3.5-turbo, deepseek-ai/DeepSeek-V3.")
    parser.add_argument("--llm_provider", type=str, default="chatgpt", choices=["chatgpt", "deepseek"],
                        help="The LLM provider to use. Available options: chatgpt, deepseek.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of triples to process in each batch (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed LLM response")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"ENTITY EXPANSION EVALUATION")
    print(f"{'='*80}")
    print(f"Entity ID: {args.entity}")
    print(f"Model: {args.model}")
    print(f"LLM Provider: {args.llm_provider}")
    print(f"Batch Size: {args.batch_size}")
    print(f"{'='*80}")
    
    # Get entity triples
    print("Retrieving entity triples...")
    entity_name, triples = get_entity_triples_for_evaluation(args.entity)
    
    if not entity_name or not triples:
        print(f"❌ Could not retrieve triples for entity {args.entity}")
        sys.exit(1)
    
    print(f"Entity Name: {entity_name}")
    print(f"Found {len(triples)} triples")
    
    # Show triples if verbose
    if args.verbose:
        print(f"\nTriples:")
        for i, triple in enumerate(triples, 1):
            print(f"  {i:2d}. [{triple[0]}, {triple[1]}, {triple[2]}]")
    
    print(f"\nEvaluating with {args.llm_provider.upper()} in batches of {args.batch_size}...")
    
    # Evaluate with LLM
    should_not_expand, llm_response, informative_triples = evaluate_entity_expansion_worthiness(
        args.entity, entity_name, triples, args.model, args.llm_provider, args.batch_size
    )
    
    if should_not_expand is None:
        print("❌ Failed to evaluate entity")
        sys.exit(1)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    
    if should_not_expand:
        print("🚫 RECOMMENDATION: Do NOT expand this entity")
        print("   All triples are non-informative or trivial")
    else:
        print("✅ RECOMMENDATION: Entity is worth expanding")
        print("   Contains informative triples")
        if informative_triples:
            print(f"\n📋 Informative triples found ({len(informative_triples)} triples):")
            for i, triple in enumerate(informative_triples, 1):
                print(f"  {i:2d}. [{triple[0]}, {triple[1]}, {triple[2]}]")
    
    if args.verbose:
        print(f"\nLLM Response:")
        print(f"{llm_response}")
    
    print(f"{'='*80}")
    
    # Return appropriate exit code
    sys.exit(0 if should_not_expand else 1)

if __name__ == "__main__":
    main()
