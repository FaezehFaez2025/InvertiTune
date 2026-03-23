import os
import json

def read_triplets_per_line(file_path):
    """Read triplets from a file where each line is a JSON array of triplets."""
    triplets_per_sample = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    triplets = json.loads(line)
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    triplets = []
                triplets_per_sample.append(triplets)
            else:
                triplets_per_sample.append([])
    return triplets_per_sample

def main():
    # Define the input directory
    input_dir = "result/controlled_extraction/test/original_prediction"

    # List of files to process (order matters: last is openie6)
    files = [
        "aggregated_chatgpt_triplets.txt",
        "aggregated_graphrag_triplets.txt",
        "aggregated_lightrag_triplets.txt",
        "aggregated_finetuned_1.5B_improved_prediction_triplets.txt",
        "aggregated_openie6_triplets.txt"
    ]

    # Read all files as lists of triplet lists
    all_samples = []
    for file in files:
        file_path = os.path.join(input_dir, file)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return
        samples = read_triplets_per_line(file_path)
        all_samples.append(samples)
        print(f"Read {len(samples)} samples from {file}")

    # Check all files have the same number of samples
    num_samples = len(all_samples[0])
    if not all(len(samples) == num_samples for samples in all_samples):
        print("Mismatch in number of samples across files!")
        return

    # Find indices where all methods have non-empty results
    valid_indices = [
        i for i in range(num_samples)
        if all(len(samples[i]) > 0 for samples in all_samples)
    ]
    print(f"Found {len(valid_indices)} samples with non-empty results from all methods")

    # Get openie6 results for these indices
    openie6_samples = all_samples[-1]

    # Create output directory (one level up from original_prediction)
    output_dir = os.path.dirname(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save filtered openie6 results to a new file (JSON per line)
    output_file = os.path.join(output_dir, "aggregated_openie6_triplets.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in valid_indices:
            f.write(json.dumps(openie6_samples[i], ensure_ascii=False) + "\n")

    print(f"Saved filtered openie6 triplets to {output_file}")

if __name__ == "__main__":
    main() 