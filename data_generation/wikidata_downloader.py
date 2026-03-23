import os
import requests
import bz2
import shutil

# Updated URL for the latest truthy RDF dump
DUMP_URL = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-lexemes.nt.bz2"
OUTPUT_FILE = "knowledge_base.nt"

def download_file(url, output_file):
    """Download a file from a URL."""
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Download complete: {output_file}")

def decompress_bz2(file_path, output_file):
    """Decompress a .bz2 file."""
    print(f"Decompressing {file_path}...")
    with bz2.BZ2File(file_path, "rb") as f_in:
        with open(output_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompression complete: {output_file}")

def main():
    try:
        # Step 1: Download the truthy dump
        compressed_file = DUMP_URL.split("/")[-1]
        download_file(DUMP_URL, compressed_file)

        # Step 2: Decompress the dump
        decompress_bz2(compressed_file, OUTPUT_FILE)

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print("The requested file may not exist or the URL has changed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
