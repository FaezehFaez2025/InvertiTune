import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iteration', type=int, required=True, help='Iteration number (1, 2, etc.)')
parser.add_argument('--output_dir', type=str, default='GPT3.5_result_KELMs', help='Output directory name')
args = parser.parse_args()

# Create main output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Create iteration directory if it doesn't exist
iteration_dir = f"{args.output_dir}/Iteration{args.iteration}"
os.makedirs(iteration_dir, exist_ok=True)

# Get number of lines from test_generated_graphs.txt
test_graphs_path = os.path.join(iteration_dir, 'test_generated_graphs.txt')
num_lines = 0
with open(test_graphs_path, 'r') as f:
    num_lines = sum(1 for _ in f)

# Create dummy files
dummy_files = {
    'train.source': 'dummy text <S> dummy graph\n',
    'train.target': 'dummy triple\n',
    'val.source': 'dummy text <S> dummy graph\n',
    'val.target': 'dummy triple\n',
    'test.target': 'dummy triple\n' * num_lines
}

for filename, content in dummy_files.items():
    filepath = os.path.join(iteration_dir, filename)
    with open(filepath, 'w') as f:
        f.write(content)

print(f"Created dummy files in {iteration_dir}/") 