import os
import argparse
import logging
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm  # Import tqdm for the progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the base prompt template
PROMPT_TEMPLATE = """
You are a triple validator. Your task is to determine if a given triple is meaningful and human-readable. The triple should clearly express a relationship between the subject, predicate, and object, and the predicate must be in English.

Guidelines:
- The triple must not contain non-human-readable data (e.g., random strings, encoded values, IDs, or URLs).
- Triples containing dates, numbers (e.g., page numbers, publication dates) are valid if they are meaningful and human-readable.
- The triple must not contain non-Latin characters (e.g., Chinese, Arabic, Persian, Russian, etc.).
- When converted into a sentence, the triple should form a meaningful and grammatically correct English statement. 

Examples of valid triples:
- ["China", "instance of", "country"]
- ["human", "has characteristic", "bipedalism"]
- ["Albert Einstein", "discovered", "theory of relativity"]
- ["Antarctica", "performer", "The Secret Handshake"]
- ["The Secret Handshake", "has part(s)", "Luis Dubuc"]
- ["Antarctica", "publication date", "2004-01-01T00:00:00Z"]
- ["Antarctica", "instance of", "album"]
- ["Belgium", "page(s)", "164"]
- ["Belgium", "publication date", "1929-07-01T00:00:00Z"]
- ["Belgium", "volume", "2"]
- ["Belgium", "instance of", "scholarly article"]

Examples of invalid triples:
- ["African Lion", "MMSI", "353884000"]
- ["human", "Krugosvet article", "biologiya/chelovek-razumnyi-homo-sapiens"]

Now, validate the following {input_type}:
{input_data}

{output_instruction}
"""

PATH_PROMPT_TEMPLATE = """
You are a path validator. Your task is to determine if a given path of triples is meaningful and useful for generating multi-hop questions.
A path is a chain of [subject, predicate, object] triples, where the object of one triple becomes the subject of the next triple (except for the first subject and the final object).

Guidelines:
- Ensure each triple in the path is human-readable, coherent, and adds clear information.
- Avoid trivial or overly general triples (e.g., ["human", "has characteristic", "self-awareness"]).
- Discard any invalid or unreadable triples.

Examples of good paths (should be accepted):
- ["Monaco", "designed by", "Susan Kare"], ["Susan Kare", "educated at", "Harriton High School"], ["Harriton High School", "country", "United States"]
- ["Edward Zwick", "place of birth", "Chicago"], ["Chicago", "located in the statistical territorial entity", "Chicago metropolitan area"], ["Chicago metropolitan area", "located in time zone", "UTC−05:00"]

Now, validate the following {input_type}:
{input_data}

{output_instruction}
"""

def validate_with_llm(input_data, model, llm_provider, mode, validation_type="triple"):
    """
    Validate triples or paths using either DeepSeek or ChatGPT.
    """
    # Define the relative path to the .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
    load_dotenv(dotenv_path)

    # Initialize the OpenAI client based on the LLM provider
    if llm_provider == "deepseek":
        client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.getenv("NEBIUS_API_KEY")
        )
    elif llm_provider == "chatgpt":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("Invalid LLM provider. Choose either 'deepseek' or 'chatgpt'.")

    # Customize the prompt based on the mode and validation type
    if validation_type == "triple":
        prompt_template = PROMPT_TEMPLATE
        if mode == "single":
            input_type = "triple"
            output_instruction = "Answer only 'Yes' or 'No'."
        else:
            input_type = "list of triples"
            output_instruction = "Return only the list of valid triples in the same format as the input."
    else:  # path validation
        prompt_template = PATH_PROMPT_TEMPLATE
        if mode == "single":
            input_type = "path"
            output_instruction = "Answer only 'Yes' or 'No'."
        else:
            input_type = "list of paths"
            output_instruction = "Return only the list of valid paths in the same format as the input."

    # Format the prompt
    prompt = prompt_template.format(
        input_type=input_type,
        input_data=input_data,
        output_instruction=output_instruction
    )

    try:
        # Call the API based on the LLM provider
        if llm_provider == "deepseek":
            response = client.chat.completions.create(
                model=model,
                max_tokens=2048 if mode == "batch" else 512,  # Increase max_tokens for batch processing
                temperature=0.0,  # No randomness for validation
                top_p=0.95,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that validates triples/paths."},
                    {"role": "user", "content": prompt}
                ]
            )
        elif llm_provider == "chatgpt":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that validates triples/paths."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0  # No randomness for validation
            )

        # Extract the response
        if mode == "single":
            validation_result = response.choices[0].message.content.strip().lower()
            return validation_result == "yes"
        elif mode == "batch":
            valid_items = response.choices[0].message.content.strip()
            return eval(valid_items)  # Convert the string response back to a list
    except Exception as e:
        logging.error(f"An error occurred while calling the {llm_provider.upper()} API: {e}")
        return False if mode == "single" else []

def process_triples_file(input_file, output_file, filtered_file, model, llm_provider, mode, batch_size=None):
    """
    Process a triples file, validate each triple using the chosen LLM, and write valid triples to the output file.
    Also, save the filtered triples (those removed) to a separate file.
    """
    valid_triples = []
    filtered_triples = []  # Track triples that are removed

    # Read the input file
    with open(input_file, "r") as file:
        triples = [line.strip() for line in file.readlines() if line.strip()]

    if mode == "single":
        # Validate each triple individually
        for triple in triples:
            if validate_with_llm(triple, model, llm_provider, mode):
                valid_triples.append(triple)
            else:
                filtered_triples.append(triple)  # Add to filtered triples
    elif mode == "batch":
        if batch_size:
            # Process triples in batches of size `batch_size`
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i + batch_size]
                valid_batch = validate_with_llm(batch, model, llm_provider, mode)
                valid_triples.extend(valid_batch)
                # Add the filtered triples from this batch
                filtered_triples.extend([triple for triple in batch if triple not in valid_batch])
        else:
            # Validate all triples in a single batch
            valid_triples = validate_with_llm(triples, model, llm_provider, mode)
            # Add the filtered triples
            filtered_triples.extend([triple for triple in triples if triple not in valid_triples])
    else:
        raise ValueError("Invalid mode. Choose either 'single' or 'batch'.")

    # Write valid triples to the output file
    with open(output_file, "w") as file:
        for triple in valid_triples:
            file.write(f"{triple}\n")

    # Write filtered triples to a separate file
    with open(filtered_file, "w") as file:
        for triple in filtered_triples:
            file.write(f"{triple}\n")

    logging.info(f"Processed {len(triples)} triples. Found {len(valid_triples)} valid triples.")

def process_paths_file(input_file, output_file, model, llm_provider, mode, batch_size=None):
    """
    Process a paths file, validate each path using the chosen LLM, and write valid paths to the output file.
    """
    valid_paths = []
    PATH_SEPARATOR = "=" * 50

    # Read and parse the input file
    with open(input_file, "r") as file:
        content = file.read()
    
    # Split into individual paths and clean empty lines
    raw_paths = [path.strip() for path in content.split(PATH_SEPARATOR) if path.strip()]
    paths = []
    for raw_path in raw_paths:
        triples = [line.strip() for line in raw_path.split("\n") if line.strip()]
        if triples:  # Only process non-empty paths
            paths.append(triples)

    # Validate paths based on processing mode
    if mode == "single":
        # Validate each path individually
        for path in paths:
            if validate_with_llm(path, model, llm_provider, mode, "path"):
                valid_paths.append(path)
    elif mode == "batch":
        if batch_size:
            # Process in batches
            for i in range(0, len(paths), batch_size):
                batch = paths[i:i + batch_size]
                valid_batch = validate_with_llm(batch, model, llm_provider, mode, "path")
                valid_paths.extend(valid_batch)
        else:
            # Single batch processing
            valid_paths = validate_with_llm(paths, model, llm_provider, mode, "path")
    else:
        raise ValueError("Invalid mode. Choose either 'single' or 'batch'.")

    # Write valid paths to output file with proper formatting
    with open(output_file, "w") as file:
        for path in valid_paths:
            file.write("\n".join(path) + "\n")
            file.write(PATH_SEPARATOR + "\n")

    logging.info(f"Processed {len(paths)} paths. Found {len(valid_paths)} valid paths.")

def process_single_file(filename, source_dir, model, llm_provider, mode, batch_size=None, validation_type="triple"):
    """
    Process a single file (either triples or paths).
    """
    if validation_type == "triple":
        if not filename.endswith("_triples.txt"):
            return
        input_file = os.path.join(source_dir, filename)
        output_file = os.path.join(source_dir, filename.replace("_triples.txt", "_triples_pruned.txt"))
        filtered_file = output_file.replace("_triples_pruned.txt", "_triples_filtered.txt")
    else:  # path validation
        if not filename.endswith("_paths.txt"):
            return
        input_file = os.path.join(source_dir, filename)
        output_file = os.path.join(source_dir, filename.replace("_paths.txt", "_paths_pruned.txt"))

    logging.info(f"Processing file: {input_file}")
    
    if validation_type == "triple":
        process_triples_file(input_file, output_file, filtered_file, model, llm_provider, mode, batch_size)
    else:
        process_paths_file(input_file, output_file, model, llm_provider, mode, batch_size)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Prune triples or paths using an LLM to filter meaningful and human-readable items.")
    parser.add_argument("--source", type=str, default="wikidata", choices=["wikidata", "yago"], help="Data source")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model to use (e.g., gpt-3.5-turbo, deepseek-ai/DeepSeek-V3)")
    parser.add_argument("--llm_provider", type=str, default="chatgpt", choices=["chatgpt", "deepseek"], help="The LLM provider to use (chatgpt or deepseek)")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "batch"], help="Processing mode: 'single' (one LLM call per item) or 'batch' (one LLM call per file/batch)")
    parser.add_argument("--batch_size", type=int, default=None, help="Number of items to process in each batch (only applicable in batch mode)")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing of files")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel processing (default: 4)")
    parser.add_argument("--validation_type", type=str, default="triple", choices=["triple", "path"], help="Whether to validate triples or paths")

    args = parser.parse_args()

    # Define the base directory
    base_dir = "./data"
    source_dir = os.path.join(base_dir, args.source)

    # Get the list of files to process based on validation type
    if args.validation_type == "triple":
        txt_files = [filename for filename in os.listdir(source_dir) if filename.endswith("_triples.txt")]
    else:
        txt_files = [filename for filename in os.listdir(source_dir) if filename.endswith("_paths.txt")]

    if args.parallel:
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = []
            for filename in txt_files:
                futures.append(executor.submit(process_single_file, filename, source_dir, args.model, args.llm_provider, args.mode, args.batch_size, args.validation_type))

            # Track progress with tqdm
            with tqdm(total=len(txt_files), desc="Processing files", unit="file") as pbar:
                for future in as_completed(futures):
                    future.result()  # Wait for the task to complete
                    pbar.update(1)
    else:
        # Sequential processing
        for filename in tqdm(txt_files, desc="Processing files", unit="file"):
            process_single_file(filename, source_dir, args.model, args.llm_provider, args.mode, args.batch_size,args.validation_type)

if __name__ == "__main__":
    main()