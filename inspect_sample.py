import json
import random
import argparse

def inspect_random_sample(jsonl_path):
    """Display a random sample's text and triples from a JSONL file"""
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]
    
    sample = random.choice(data)
    
    print(f"\n=== Random Sample ===")
    print(f"\nText:\n{sample['text']}")
    print(f"\nGround Truth Triples:\n{json.dumps(sample['ground_truth_triples'], indent=2)}")
    print(f"\nGenerated Triples:\n{sample['generated_triples']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect a random sample from JSONL')
    parser.add_argument('jsonl_file', help='Path to JSONL file')
    args = parser.parse_args()
    inspect_random_sample(args.jsonl_file)