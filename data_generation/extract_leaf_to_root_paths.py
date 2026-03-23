import os
import argparse
from graph import GraphNode

def extract_leaf_to_root_paths(json_file, source_dir, postfix):
    """Extracts paths from leaves to root."""
    json_file_path = os.path.join(source_dir, json_file)
    root_node = GraphNode.load_graph(json_file_path)

    # Get leaves as node objects
    leaves = root_node.get_leaves()
    
    # Prepare output file path
    paths_file = json_file.replace(postfix, "_paths.txt")
    paths_file_path = os.path.join(source_dir, paths_file)
    
    with open(paths_file_path, 'w') as f:
        for i, leaf_node in enumerate(leaves):
            paths = leaf_node.get_path()
            
            # Write all paths for this leaf
            for path in paths:
                for triple in path:
                    # Write in the format ["subject", "predicate", "object"]
                    f.write(f'["{triple[0]}", "{triple[1]}", "{triple[2]}"]\n')
                if len(paths) > 1:  # Only add separator if multiple paths
                    f.write("--------\n")
            
            # Add leaf separator (except after last leaf)
            if i < len(leaves) - 1:
                f.write("==================================================\n")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract all paths from leaf nodes to the root in knowledge graphs."
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="wikidata", 
        choices=["wikidata", "yago"], 
        help="Data source (wikidata or yago)"
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default="_triples.json",
        help="Postfix pattern for graph files (e.g., '_triples_pruned.json')"
    )
    args = parser.parse_args()

    # Define the base directory
    base_dir = "./data"
    source_dir = os.path.join(base_dir, args.source)

    # Get all graph files matching the postfix
    json_files = [
        filename for filename in os.listdir(source_dir) 
        if filename.endswith(args.postfix)
    ]

    if not json_files:
        print(f"No graph files with postfix '{args.postfix}' found in: {source_dir}")
        return

    # Process each graph file
    for json_file in json_files:
        extract_leaf_to_root_paths(json_file, source_dir, args.postfix)
        print(f"Saved paths for {json_file} to {json_file.replace(args.postfix, '_paths.txt')}")

if __name__ == "__main__":
    main()