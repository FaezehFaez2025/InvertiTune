import os
import json
import argparse

from find_common_triples import read_predictions, is_empty_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/original_prediction/aggregated_pive_triplets.txt",
        help="Input file to sanitize (relative path).",
    )
    parser.add_argument(
        "--output",
        default="results/original_prediction/aggregated_pive_triplets_sanitized.txt",
        help="Output file to write sanitized predictions (relative path).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Use the same parsing logic as find_common_triples.py
    predictions = read_predictions(args.input)
    if predictions is None:
        raise RuntimeError(f"Failed to read predictions from {input_path}")

    num_lines = len(predictions)
    num_fixed = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for pred in predictions:
            if is_empty_prediction(pred):
                num_fixed += 1
                cleaned = [["", "", ""]]
            else:
                cleaned = pred
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    print(f"Processed {num_lines} lines.")
    print(f"Replaced {num_fixed} unparsable or empty lines with [['', '', '']].")
    print(f"Sanitized file written to: {args.output}")


if __name__ == "__main__":
    main()

