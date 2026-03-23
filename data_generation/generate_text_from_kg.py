import os
import argparse
import logging
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm  # Import tqdm for the progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_text_from_knowledge_graph(triples, model, llm_provider):
    """
    Generate a textual description from a list of triples (knowledge graph) using the specified model.
    The goal is to reconstruct the original text that could have generated the given knowledge graph.
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

    prompt = """
            You are a text generator that reconstructs the original text from a given knowledge graph. The knowledge graph is represented as a list of triples in the format: ["subject", "relation", "object"].

            Your task is to generate a coherent, concise, and natural text that could have been the origin of the given knowledge graph. The text should accurately describe the relationships and entities in the triples, ensuring it is informative and logically structured.

            Guidelines:
            1. The generated text can consist of one or more paragraphs, depending on the complexity of the triples.
            2. Ensure the text flows naturally, as if it were written by a human.
            3. Include all entities and relationships from the triples.
            4. Avoid adding any information not present in the triples.

            Triples:
            {}
            Text:
            """.format(triples)

    try:
        # Call the API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that reconstructs original text from knowledge graphs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # Slightly higher temperature for creativity
        )
        # Extract the generated text
        generated_text = response.choices[0].message.content
        return generated_text
    except Exception as e:
        logging.error(f"An error occurred while calling the {llm_provider.upper()} API: {e}")
        return None

def process_file(file, source_dir, model, llm_provider, postfix):
    """Process a single file to generate text from its triples"""
    input_file_path = os.path.join(source_dir, file)
    # Generate output filename by replacing the postfix with '_text.txt'
    output_file_path = os.path.join(source_dir, file.replace(postfix, "_text.txt"))

    # Read the input triples from the input file
    try:
        with open(input_file_path, "r") as file:
            input_triples = file.read()
    except FileNotFoundError:
        logging.error(f"The input file '{input_file_path}' does not exist.")
        return

    # Generate the textual description
    generated_text = generate_text_from_knowledge_graph(input_triples, model, llm_provider)

    if generated_text:
        # Save the output to the output file
        try:
            with open(output_file_path, "w") as f:
                f.write(generated_text)
            logging.info(f"Generated text saved to {output_file_path}")
        except Exception as e:
            logging.error(f"An error occurred while saving the output file: {e}")
    else:
        logging.error(f"Failed to generate text from knowledge graph for {input_file_path}.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a textual description from a knowledge graph represented as triples.")
    parser.add_argument("--source", type=str, default="wikidata", choices=["wikidata", "yago"],
                        help="Source of the data (wikidata or yago). Default is 'wikidata'.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="The model to use. Available options: gpt-4o, gpt-4o-mini, gpt-3.5-turbo, deepseek-ai/DeepSeek-V3.")
    parser.add_argument("--llm_provider", type=str, default="chatgpt", choices=["chatgpt", "deepseek"],
                        help="The LLM provider to use. Available options: chatgpt, deepseek.")
    parser.add_argument("--postfix", type=str, default="_triples_pruned.txt",
                        help="Postfix of the files to be processed. Default is '_triples_pruned.txt'.")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of threads for processing. Default is 1 (sequential processing).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="If set, only generate text for files whose corresponding _text.txt file does not exist.")

    args = parser.parse_args()

    # Determine the directory based on the source
    base_dir = "data"
    source_dir = os.path.join(base_dir, args.source)

    # Check if the directory exists
    if not os.path.exists(source_dir):
        logging.error(f"Directory {source_dir} does not exist.")
        return

    # Get all files ending with the specified postfix
    files = [f for f in os.listdir(source_dir) if f.endswith(args.postfix)]

    # Filter out files that already have an output if --skip_existing is provided
    if args.skip_existing:
        files = [f for f in files if not os.path.exists(os.path.join(source_dir, f.replace(args.postfix, "_text.txt")))]
        if not files:
            logging.info("No new files to process. All files already have corresponding _text.txt outputs.")
            return

    if not files:
        logging.error(f"No files ending with '{args.postfix}' found in {source_dir}.")
        return

    # Print the number of files to be processed
    print(f"\nNumber of files to process: {len(files)}")
    print("Starting processing...\n")

    if args.num_threads > 1:
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = []
            for file in files:
                futures.append(executor.submit(
                    process_file, 
                    file, 
                    source_dir, 
                    args.model, 
                    args.llm_provider, 
                    args.postfix
                ))
            
            # Track progress with tqdm
            for _ in tqdm(as_completed(futures), total=len(files), desc="Processing files", unit="file"):
                pass
    else:
        # Process each file sequentially with a progress bar
        for file in tqdm(files, desc="Processing files", unit="file"):
            process_file(file, source_dir, args.model, args.llm_provider, args.postfix)

if __name__ == "__main__":
    main()