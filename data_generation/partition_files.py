import os
import argparse
import tarfile
import shutil
from pathlib import Path

def partition_files(source: str, num_partitions: int):
    # Determine the base directory
    base_dir = Path("data") / source
    if not base_dir.exists():
        raise FileNotFoundError(f"The directory {base_dir} does not exist.")

    # Gather all files ending with _triples.txt
    files = list(base_dir.glob("*_triples.txt"))
    if not files:
        raise FileNotFoundError(f"No files ending with '_triples.txt' found in {base_dir}.")

    # Calculate the number of files per partition
    total_files = len(files)
    base_files_per_partition = total_files // num_partitions
    remainder = total_files % num_partitions

    # Create a partitions folder in the same directory as the data folder
    partitions_dir = Path("partitions")
    if partitions_dir.exists():
        # Remove the existing partitions folder and its contents
        shutil.rmtree(partitions_dir)
    partitions_dir.mkdir()

    # Create partitions
    start_index = 0
    for i in range(num_partitions):
        # Calculate the number of files for this partition
        if i < remainder:
            files_in_partition = base_files_per_partition + 1
        else:
            files_in_partition = base_files_per_partition

        # Get the files for this partition
        partition_files = files[start_index:start_index + files_in_partition]
        start_index += files_in_partition

        # Create a folder for this partition
        partition_folder_name = f"{source}_partition_{i + 1}"
        partition_folder_path = partitions_dir / partition_folder_name
        partition_folder_path.mkdir()

        # Copy files into the partition folder
        for file in partition_files:
            shutil.copy(file, partition_folder_path)

        # Create a .tar.gz archive for this partition
        partition_name = f"{partition_folder_name}.tar.gz"
        partition_path = partitions_dir / partition_name
        with tarfile.open(partition_path, "w:gz") as tar:
            tar.add(partition_folder_path, arcname=partition_folder_name)

        # Remove the partition folder after creating the archive
        shutil.rmtree(partition_folder_path)

        print(f"Created {partition_name} with {len(partition_files)} files.")

def main():
    parser = argparse.ArgumentParser(description="Partition files ending with '_triples.txt' into equally sized .tar.gz archives.")
    parser.add_argument("--source", type=str, required=True, choices=["wikidata", "yago"],
                        help="The source folder inside 'data' (e.g., 'wikidata' or 'yago').")
    parser.add_argument("--num_partitions", type=int, required=True,
                        help="The number of partitions to create.")
    args = parser.parse_args()

    try:
        partition_files(args.source, args.num_partitions)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()