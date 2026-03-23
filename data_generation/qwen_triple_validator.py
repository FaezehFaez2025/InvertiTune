import os
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROMPT_TEMPLATE = """
You are a triple validator. Your task is to determine if a given triple is meaningful and human-readable. The triple should clearly express a relationship between the subject, predicate, and object, 
and the predicate must be in English.

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
- ["African Lion", "MMSI", "353884000"] (contains a technical ID)
- ["human", "Google News topics ID", "CAAqJggKIiBDQkFTRWdvSkwyMHZNR1JuZHpseUVnVmxiaTFIUWlnQVAB"] (contains a technical ID)
- ["human", "pronunciation audio", "http://commons.wikimedia.org/wiki/Special:FilePath/LL-Q13955%20%28ara%29-Spotless%20Mind1988-%D8%A5%D9%86%D8%B3%D8%A7%D9%86.wav"] (contains a URL and non-English 
text)
- ["human", "Krugosvet article", "biologiya/chelovek-razumnyi-homo-sapiens"] (contains non-English text)

Now, validate the following triple:
{triple}

Answer only 'Yes' or 'No'.
"""

def load_qwen_model(num_params: str):
    """
    Load the local Qwen model and tokenizer, and return them along with the device.
    """
    model_name = f"Qwen/Qwen2.5-{num_params}-Instruct"
    logging.info(f"Loading Qwen model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="<pad>")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

def validate_triple_with_qwen(triple_str: str, model, tokenizer, device: str) -> bool:
    """
    Given a single triple (string form), use the local Qwen model to determine
    if the triple is valid ("Yes") or not ("No").
    Returns True if "Yes", otherwise False.
    """
    # Fill the prompt
    prompt = PROMPT_TEMPLATE.format(triple=triple_str)

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that validates triples."},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=512, temperature=0.7, top_p=0.8, repetition_penalty=1.05)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the last line of the generated text
    last_line = generated_text.strip().split('\n')[-1].strip()

    # Return True if the last line suggests "Yes"
    lower_answer = last_line.lower()
    return ("yes" in lower_answer) and ("no" not in lower_answer)

def process_triples_file(input_file: str, output_file: str, model, tokenizer, device: str):
    """
    Process a single file of triples line by line. If Qwen says "Yes", we keep the triple.
    """
    valid_triples = []
    with open(input_file, "r", encoding="utf-8") as f_in:
        triples = [line.strip() for line in f_in if line.strip()]

    for triple in tqdm(triples, desc=f"Processing triples in {os.path.basename(input_file)}", unit="triple"):
        if validate_triple_with_qwen(triple, model, tokenizer, device):
            valid_triples.append(triple)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for vt in valid_triples:
            f_out.write(vt + "\n")

    logging.info(f"Processed {len(triples)} triples from {input_file}. Valid: {len(valid_triples)}.")

def main():
    parser = argparse.ArgumentParser(description="Prune triples in '_triples.txt' files using a local Qwen model.")
    parser.add_argument("--source", type=str, default="wikidata", choices=["wikidata", "yago"],
                        help="Select which data source folder to process (e.g., 'wikidata' or 'yago').")
    parser.add_argument("--num_params", type=str, required=True,
                        help="Number of parameters for Qwen (e.g. '1.5B', '7B', '14B').")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of threads to use for processing files.")
    args = parser.parse_args()

    # Load the Qwen model
    model, tokenizer, device = load_qwen_model(args.num_params)

    # Base directory and source folder
    base_dir = "./data"
    source_dir = os.path.join(base_dir, args.source)

    # Gather all *_triples.txt files
    txt_files = [f for f in os.listdir(source_dir) if f.endswith("_triples.txt")]

    # Process each file concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for filename in txt_files:
            input_file = os.path.join(source_dir, filename)
            output_file = os.path.join(source_dir, filename.replace("_triples.txt", "_triples_pruned.txt"))
            futures.append(executor.submit(process_triples_file, input_file, output_file, model, tokenizer, device))

        # Track progress of file processing
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            future.result()  # Wait for each task to complete

if __name__ == "__main__":
    main()