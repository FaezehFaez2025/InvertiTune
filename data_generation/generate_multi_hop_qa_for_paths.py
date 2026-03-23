import os
import json
import logging
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

QUESTION_ANSWER_PROMPT_TEMPLATE = """
You are a question-answer generator. Given a valid path of triples, create:
1. One multi-hop question that can only be answered by following the chain of information in the path
2. The correct answer to the question

Path format: 
[subject, predicate, object], [subject, predicate, object], ...

Guidelines:
- The question must require using information from at least two triples in the path
- The question should be concise, clear, and grammatically correct
- The answer should be a short factual statement
- Both question and answer must be in English

Example:
Path:
["Monaco", "designed by", "Susan Kare"], ["Susan Kare", "educated at", "Harriton High School"], ["Harriton High School", "country", "United States"]

Output:
Question: Which country is the high school attended by the person who designed Monaco located in?
Answer: United States
"""

def create_llm_client(llm_provider, model):
    """
    Create and return an OpenAI (or DeepSeek) client.
    """
    dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
    load_dotenv(dotenv_path)

    if llm_provider == "deepseek":
        return OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.getenv("NEBIUS_API_KEY")
        )
    elif llm_provider == "chatgpt":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("Invalid LLM provider. Choose either 'deepseek' or 'chatgpt'.")

def generate_qa_pair(path, client, model, llm_provider):
    """
    Generate a question-answer pair for a single path using the LLM client.
    Returns a tuple (question, answer) or (None, None) on failure
    """
    path_str = ", ".join(str(triple) for triple in path)
    prompt = QUESTION_ANSWER_PROMPT_TEMPLATE + f"\nPath:\n{path_str}\n\nOutput:"

    try:
        if llm_provider == "deepseek":
            response = client.chat.completions.create(
                model=model,
                max_tokens=512,
                temperature=0.0,
                top_p=0.95,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates multi-hop QA from path of triples."},
                    {"role": "user", "content": prompt}
                ]
            )
        elif llm_provider == "chatgpt":
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates multi-hop QA from path of triples."},
                    {"role": "user", "content": prompt}
                ]
            )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse question and answer from response
        if "Question:" in response_text and "Answer:" in response_text:
            question_part, answer_part = response_text.split("Answer:", 1)
            question = question_part.replace("Question:", "").strip()
            answer = answer_part.strip()
            return question, answer
        
        logging.warning(f"Unexpected response format for path: {path_str}")
        return None, None

    except Exception as e:
        logging.error(f"Error generating QA pair: {e}")
        return None, None

def process_single_path_file(filepath, model, llm_provider):
    """
    Process a paths file and save QA pairs with answers
    """
    PATH_SEPARATOR = "=" * 50
    qa_records = []

    # Load and parse paths
    with open(filepath, "r") as f:
        content = f.read()

    raw_paths = [block.strip() for block in content.split(PATH_SEPARATOR) if block.strip()]
    
    paths = []
    for raw_path in raw_paths:
        lines = [line.strip() for line in raw_path.split("\n") if line.strip()]
        if lines:
            paths.append(lines)

    # Create LLM client
    client = create_llm_client(llm_provider, model)

    # Generate QA pairs
    for path in tqdm(paths, desc=f"Processing {os.path.basename(filepath)}", unit="path"):
        question, answer = generate_qa_pair(path, client, model, llm_provider)
        
        if question and answer:
            qa_records.append({
                "path": path,
                "question": question,
                "answer": answer
            })

    # Save results
    output_file = filepath.replace("_paths.txt", "_qa_pairs.json")
    with open(output_file, "w") as f:
        json.dump({
            "count": len(qa_records),
            "qa_pairs": qa_records
        }, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved {len(qa_records)} QA pairs to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-hop QA pairs from paths."
    )
    parser.add_argument(
        "--source", type=str, default="wikidata",
        choices=["wikidata", "yago"],
        help="Data source directory under ./data"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="LLM model (e.g. 'gpt-3.5-turbo' or 'deepseek-ai/DeepSeek-V3')"
    )
    parser.add_argument(
        "--llm_provider", type=str, default="chatgpt",
        choices=["chatgpt", "deepseek"],
        help="Which LLM provider to use"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Enable parallel processing of files"
    )
    parser.add_argument(
        "--num_threads", type=int, default=4,
        help="Number of threads for parallel processing (default: 4)"
    )

    args = parser.parse_args()

    # Base directory where data resides
    base_dir = "./data"
    source_dir = os.path.join(base_dir, args.source)

    # Find all *_paths.txt files
    path_files = [
        f for f in os.listdir(source_dir)
        if f.endswith("_paths.txt")
    ]

    if not path_files:
        logging.info(f"No path files found in '{source_dir}'.")
        return

    # Full paths for each file
    path_files = [os.path.join(source_dir, f) for f in path_files]

    if args.parallel:
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = []
            for file in path_files:
                futures.append(executor.submit(
                    process_single_path_file,
                    file,
                    args.model,
                    args.llm_provider
                ))

            # Track progress
            with tqdm(total=len(path_files), desc="Overall progress", unit="file") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)
    else:
        # Sequential processing
        for file in tqdm(path_files, desc="Overall progress", unit="file"):
            process_single_path_file(file, args.model, args.llm_provider)

if __name__ == "__main__":
    main()