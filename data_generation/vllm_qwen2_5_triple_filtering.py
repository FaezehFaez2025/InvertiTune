#!/usr/bin/env python3

import os
import sys
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Define the prompt template
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

def process_triples_file(input_file: str, output_file: str, llm, sampling_params):
    """
    Process a single file of triples line by line. If the LLM says "Yes", we keep the triple.
    """
    valid_triples = []
    with open(input_file, "r", encoding="utf-8") as f_in:
        triples = [line.strip() for line in f_in if line.strip()]

    for triple in tqdm(triples, desc=f"Processing triples in {os.path.basename(input_file)}", unit="triple"):
        # Format the prompt using the PROMPT_TEMPLATE
        prompt = PROMPT_TEMPLATE.format(triple=triple)

        # vLLM’s LLM.generate accepts a *list* of prompts
        result = llm.generate([prompt], sampling_params)

        # Extract the response (Yes/No)
        generated_text = result.generations[0].text.strip().lower()
        if "yes" in generated_text and "no" not in generated_text:
            valid_triples.append(triple)

    # Save the valid triples to the output file
    with open(output_file, "w", encoding="utf-8") as f_out:
        for vt in valid_triples:
            f_out.write(vt + "\n")

    print(f"Processed {len(triples)} triples from {input_file}. Valid: {len(valid_triples)}.")

def main():
    # 1) Load the model
    llm = LLM(model="Qwen/Qwen1.5-1.8B")

    # 2) Configure your generation parameters (temperature, top_p, max_tokens, etc.)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=128,
    )

    # 3) Define the folder containing the triples files
    folder_path = "./data/wikidata"  # Replace with the path to your folder

    # 4) Gather all *_triples.txt files
    txt_files = [f for f in os.listdir(folder_path) if f.endswith("_triples.txt")]

    # 5) Process each file
    for filename in txt_files:
        input_file = os.path.join(folder_path, filename)
        output_file = os.path.join(folder_path, filename.replace("_triples.txt", "_triples_pruned.txt"))
        process_triples_file(input_file, output_file, llm, sampling_params)

if __name__ == "__main__":
    sys.exit(main())