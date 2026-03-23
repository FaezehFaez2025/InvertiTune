import os
import argparse
import json
import shutil
from graph import GraphNode  # Import the GraphNode class and its methods

def load_filtered_triples(filtered_file):
    """
    Load the filtered triples from a .txt file.
    Each line in the file is a triple in the format ["subject", "predicate", "object"].
    """
    filtered_triples = []
    with open(filtered_file, "r") as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # Convert the line (string) to a tuple
                triple = eval(line.strip())
                filtered_triples.append(tuple(triple))
    return filtered_triples

def process_graph_file(json_file, source_dir):
    """
    Process a single graph file:
    1. Load the graph from the JSON file.
    2. Load the corresponding filtered triples from the _filtered.txt file.
    3. Remove the filtered triples from the graph.
    4. Save the pruned graph to a new JSON file.
    """
    # Construct the path to the filtered triples file
    filtered_file = json_file.replace("_triples.json", "_triples_filtered.txt")
    filtered_file_path = os.path.join(source_dir, filtered_file)

    # Construct input and output paths
    json_file_path = os.path.join(source_dir, json_file)
    pruned_json_file = json_file.replace("_triples.json", "_triples_pruned.json")
    pruned_json_path = os.path.join(source_dir, pruned_json_file)

    # Check if the filtered file exists
    if not os.path.exists(filtered_file_path):
        # Just copy the original file to pruned version
        shutil.copyfile(json_file_path, pruned_json_path)
        print(f"No filtered file found. Copied original to: {pruned_json_path}")
        return

    # Load the filtered triples
    filtered_triples = load_filtered_triples(filtered_file_path)
    if not filtered_triples:
        # Just copy the original file to pruned version
        shutil.copyfile(json_file_path, pruned_json_path)
        print(f"No filtered triples found. Copied original to: {pruned_json_path}")
        return

    root_node = GraphNode.load_graph(json_file_path)
    GraphNode.synchronize_graph_with_filtered_triples(root_node, filtered_triples)
    root_node.save_graph(pruned_json_path)
    print(f"Pruned graph saved to: {pruned_json_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Prune graphs by removing filtered triples.")
    parser.add_argument("--source", type=str, default="wikidata", choices=["wikidata", "yago"], help="Data source (wikidata or yago)")
    args = parser.parse_args()

    # Define the base directory
    base_dir = "./data"
    source_dir = os.path.join(base_dir, args.source)

    # Get the list of _triples.json files in the source directory
    json_files = [filename for filename in os.listdir(source_dir) if filename.endswith("_triples.json")]

    if not json_files:
        print(f"No _triples.json files found in: {source_dir}")
        return

    # Process each JSON file
    for json_file in json_files:
        print(f"Processing file: {json_file}")
        process_graph_file(json_file, source_dir)

if __name__ == "__main__":
    main()