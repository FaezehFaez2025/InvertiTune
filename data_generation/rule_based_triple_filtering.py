import os
import argparse
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PREDICATE_BLACKLIST = {
    "Wolfram Language entity code",
    "Wolfram Language unit code",
    "Wikidata property",
    "on focus list of Wikimedia project",
    "Commons category",
    "has part(s) of the class",
    "properties for this type",
    "described by source",
}

def contains_unwanted_characters(text: str) -> bool:
    """
    Check if the text contains unwanted characters.
    Returns True if unwanted characters are found, otherwise False.
    """
    # Define Unicode ranges for unwanted characters
    unwanted_ranges = [
        (0x4E00, 0x9FFF),  # Chinese characters
        (0x0600, 0x06FF),  # Arabic characters
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        (0x0670, 0x06FF),  # Persian characters (Arabic script with additional letters)
        (0x0400, 0x04FF),  # Cyrillic characters
        (0x0500, 0x052F),  # Cyrillic Supplement
        (0x3100, 0x312F),  # Bopomofo characters
        (0x30A0, 0x30FF),  # Katakana characters
        (0x0370, 0x03FF),  # Greek and Coptic characters
        (0x1F00, 0x1FFF),  # Greek Extended characters
        (0x0980, 0x09FF),  # Bengali characters
        (0x0590, 0x05FF),  # Hebrew characters
    ]
    
    for char in text:
        for start, end in unwanted_ranges:
            if start <= ord(char) <= end:
                return True
    return False

def filter_triple(triple: str) -> bool:
    """
    Filter a triple based on the following rules:
    1. If the predicate is in the blacklist, filter the triple.
    2. If the predicate contains the word "ID", filter the triple.
    3. If the object contains "http://" or "https://", filter the triple.
    4. If the triple contains unwanted characters, filter the triple.
    5. If the subject or object contains "Category:", "Template:", "Wikipedia:", or "Portal:" filter the triple.
    6. If the subject or object starts with "Q" followed by at least 5 digits, filter the triple.
    7. If the subject and object are equal, filter the triple.
    Returns True if the triple should be kept, otherwise False.
    """
    try:
        # Split the triple by commas and extract the predicate
        predicate = triple.split(",")[1].strip().strip('"')
        
        # Rule 1: Filter if predicate is in the blacklist
        if predicate in PREDICATE_BLACKLIST:
            return False

        # Parse the triple (assuming it's in the format ["subject", "predicate", "object"])
        subject, predicate, object_ = eval(triple)

        # Rule 2: Filter if predicate contains "ID"
        if "ID" in predicate.split():
            return False

        # Rule 3: Filter if object contains "http://" or "https://"
        if "http://" in object_ or "https://" in object_:
            return False

        # Rule 4: Filter if the triple contains unwanted characters
        if (contains_unwanted_characters(subject) or
            contains_unwanted_characters(predicate) or
            contains_unwanted_characters(object_)):
            return False

        # Rule 5: Filter if the subject or object contains "Category:", "Template:", "Wikipedia:", or "Portal:"
        prefixes = ("Category:", "Template:", "Wikipedia:", "Portal:")
        if any(subject.startswith(prefix) or object_.startswith(prefix) for prefix in prefixes):
            return False

        # Rule 6: Filter if the subject or object starts with "Q" followed by at least 5 digits
        if re.match(r"^Q\d{5,}", subject) or re.match(r"^Q\d{5,}", object_):
            return False

        # Rule 7: Filter if subject and object are equal
        if subject == object_:
            return False

        # If none of the rules are violated, keep the triple
        return True

    except Exception as e:
        logging.error(f"Error parsing triple: {triple}. Error: {e}")
        return False

def process_triples_file(input_file: str, output_file: str):
    """
    Process a single file of triples line by line and apply rule-based filtering.
    """
    valid_triples = []
    with open(input_file, "r", encoding="utf-8") as f_in:
        triples = [line.strip() for line in f_in if line.strip()]

    for triple in triples:
        if filter_triple(triple):
            valid_triples.append(triple)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for vt in valid_triples:
            f_out.write(vt + "\n")

    logging.info(f"Processed {len(triples)} triples from {input_file}. Valid: {len(valid_triples)}.")

def main():
    parser = argparse.ArgumentParser(description="Prune triples in '_triples.txt' files using rule-based filtering.")
    parser.add_argument("--source", type=str, default="wikidata", choices=["wikidata", "yago"],
                        help="Select which data source folder to process (e.g., 'wikidata' or 'yago').")
    args = parser.parse_args()

    # Base directory and source folder
    base_dir = "./data"
    source_dir = os.path.join(base_dir, args.source)

    # Gather all *_triples.txt files
    txt_files = [f for f in os.listdir(source_dir) if f.endswith("_triples.txt")]

    # Process each file
    for filename in txt_files:
        input_file = os.path.join(source_dir, filename)
        output_file = os.path.join(source_dir, filename.replace("_triples.txt", "_triples_rule_based_pruned.txt"))
        process_triples_file(input_file, output_file)

if __name__ == "__main__":
    main()