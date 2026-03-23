import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--iteration', type=int, required=True, help='Iteration number (1, 2, etc.)')
parser.add_argument('--output_dir', type=str, default='GPT3.5_result_KELMs', help='Output directory name')
args = parser.parse_args()

# Create single verifier input
test_generated_graphs = []
test_texts = []
with open(f"{args.output_dir}/Iteration{args.iteration}/test_generated_graphs.txt", 'r') as f:
    for line in f.readlines():
        test_generated_graphs.append(line.strip())

with open("GPT3.5_result_KELM/test.target", 'r') as f:
    for line in f.readlines():
        test_texts.append(line.strip())

with open(f"{args.output_dir}/Iteration{args.iteration}/test.source", 'w') as f:
    for i in range(len(test_generated_graphs)):
        f.write(test_texts[i] + ' <S> ' + test_generated_graphs[i] + '\n')

print(f"Processing complete. Output written to {args.output_dir}/Iteration{args.iteration}/test.source") 