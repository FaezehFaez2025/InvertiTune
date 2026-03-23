import os
import shutil
import subprocess
import re
import json
import signal
import time

def clean_directories():
    """Remove output and result directories if they exist."""
    dirs_to_clean = ['output', 'result']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name} directory")

def process_deepex_output(output_file):
    """Process DeepEx output file and convert to required format."""
    triples = []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if len(lines) < 2:  # Need at least sentence + one triple
            return []
            
        # Skip the first line (original sentence)
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Parse tab-separated format: ID\t"Subject"\t"Relation"\t"Object"\tScore
            parts = line.split('\t')
            if len(parts) >= 4:
                # Remove quotes from subject, relation, and object
                subject = parts[1].strip('"')
                relation = parts[2].strip('"')
                obj = parts[3].strip('"')
                
                # Convert to required format
                triple = [subject.upper(), relation, obj.upper()]
                triples.append(triple)
                
    except Exception as e:
        print(f"Error processing DeepEx output: {e}")
        return []
    
    return triples

def process_sample(sample_num, test_dir, k_value=1):
    """Process a single sample."""
    print(f"\nProcessing sample {sample_num} with K={k_value}...")
    
    # Clean directories
    clean_directories()
    
    # Source and destination paths
    source_text = os.path.join(test_dir, str(sample_num), "text.txt")
    dest_dir = "supervised-oie/external_datasets/mesquita_2013/processed"
    dest_file = os.path.join(dest_dir, "web.raw")
    
    # Create parent directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy text.txt to destination
    try:
        with open(source_text, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
        
        # Split text into sentences (more robust approach)
        sentences = []
        
        # More comprehensive abbreviation list
        abbreviations = {
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'i.e', 'e.g',
            'Inc', 'Ltd', 'Co', 'Corp', 'St', 'Ave', 'Blvd', 'Rd', 'Jan', 'Feb', 
            'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
            'U.S', 'U.K', 'Ph.D', 'M.D', 'B.A', 'M.A', 'LLC', 'P.O'
        }
        
        def _is_abbreviation_ending(text_part, abbreviations, ending):
            # Don't split on multiple punctuation (ellipses, etc.)
            if len(ending) > 1:
                return True
                
            # Check if the last word is a known abbreviation
            words = text_part.split()
            if words:
                last_word = words[-1].rstrip('.,!?')
                if last_word in abbreviations:
                    return True
                    
                # Check for decimal numbers (e.g., "3.50")
                if re.match(r'\d+\.\d+$', last_word):
                    return True
                    
                # Check for single letter followed by period (e.g., "A.")
                if re.match(r'^[A-Z]$', last_word) and ending == '.':
                    return True
            
            return False
        
        # Split by sentence endings but keep the punctuation
        parts = re.split(r'([.!?]+)', text_content)
        
        current_sentence = ""
        
        for i in range(0, len(parts), 2):
            if i >= len(parts):
                break
                
            sentence_part = parts[i].strip()
            ending = parts[i + 1] if i + 1 < len(parts) else ""
            
            if sentence_part:
                current_sentence += sentence_part + ending
                
                # Check if this should end the sentence
                if ending and not _is_abbreviation_ending(sentence_part, abbreviations, ending):
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add any remaining text as a sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Clean up sentences (remove empty ones and strip whitespace)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Write each sentence to a new line in the destination file
        with open(dest_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        
        # Also save a copy to the sample folder
        sample_sentences_file = os.path.join(test_dir, str(sample_num), "sentences.txt")
        with open(sample_sentences_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        
        print(f"Copied {len(sentences)} sentences from text.txt to {dest_file}")
        print(f"Also saved sentence-split text to {sample_sentences_file}")
    except Exception as e:
        print(f"Error processing text.txt: {e}")
        return False
    
    # Run DeepEx
    try:
        # Set environment variable for K value
        env = os.environ.copy()
        env['DEEPEX_K_VALUE'] = str(k_value)
        subprocess.run(['bash', 'tasks/WEB.sh'], check=True, env=env)
        print("DeepEx processing completed")
    except subprocess.CalledProcessError as e:
        print(f"Error running DeepEx: {e}")
        return False
    
    # Process DeepEx output - use the K value in the filename
    deepex_output = f"supervised-oie/supervised-oie-benchmark/systems_output/deepex.web.{k_value}.txt"
    triples = process_deepex_output(deepex_output)
    
    # Save triples to deepex.txt in the sample folder
    output_file = os.path.join(test_dir, str(sample_num), "deepex.txt")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(triples, ensure_ascii=False) + '\n')
        print(f"Saved {len(triples)} triples to {output_file}")
    except Exception as e:
        print(f"Error saving triples: {e}")
        return False
    
    return True

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Processing timeout")

def main():
    import sys
    
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python process_samples.py <number_of_samples> <test_directory> [k_value]")
        print("  number_of_samples: Number of samples to process")
        print("  test_directory: Path to the test directory containing numbered folders")
        print("  k_value: Optional K value for top-K extraction (default: 1)")
        sys.exit(1)
    
    try:
        num_samples = int(sys.argv[1])
    except ValueError:
        print("Error: number_of_samples must be an integer")
        sys.exit(1)
    
    # Get test directory path
    test_dir = sys.argv[2]
    if not os.path.isdir(test_dir):
        print(f"Error: test directory '{test_dir}' does not exist or is not a directory")
        sys.exit(1)
    
    # Get K value (default to 1 if not provided)
    k_value = 1
    if len(sys.argv) == 4:
        try:
            k_value = int(sys.argv[3])
            if k_value < 1:
                print("Error: k_value must be a positive integer")
                sys.exit(1)
        except ValueError:
            print("Error: k_value must be an integer")
            sys.exit(1)
    
    print(f"Processing {num_samples} samples with K={k_value}...")
    print(f"Using test directory: {test_dir}")
    
    # Set timeout to 20 minutes (1200 seconds)
    timeout_seconds = 1200
    
    # Counter for failed samples
    failed_samples = 0

    timing_file = os.path.join(test_dir, "inference_times.txt")

    # Process each sample
    for i in range(1, num_samples + 1):
        elapsed_time = None
        try:
            # Set up timeout signal
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            start_time = time.time()
            success = process_sample(i, test_dir, k_value)
            elapsed_time = time.time() - start_time
            
            # Cancel the alarm
            signal.alarm(0)
            
            if not success:
                print(f"Failed to process sample {i}")
                failed_samples += 1
                continue
            else:
                print(f"Successfully processed sample {i} in {elapsed_time:.2f} seconds")
                
        except TimeoutError:
            signal.alarm(0)  # Cancel the alarm
            print(f"Timeout ({timeout_seconds}s) exceeded for sample {i}, skipping to next sample")
            failed_samples += 1
            continue
        except Exception as e:
            signal.alarm(0)  # Cancel the alarm
            print(f"Unexpected error processing sample {i}: {e}")
            failed_samples += 1
            continue
        finally:
            try:
                with open(timing_file, 'a', encoding='utf-8') as f:
                    f.write(f"sample {i}: {round(elapsed_time, 4)} seconds\n")
            except Exception as e:
                print(f"Warning: could not write timing for sample {i}: {e}")

    print(f"\nProcessing completed!")
    print(f"Total samples processed: {num_samples}")
    print(f"Successfully processed: {num_samples - failed_samples}")
    print(f"Failed to process: {failed_samples}")

if __name__ == "__main__":
    main() 