import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description='Test fine-tuned Qwen2.5 model')
parser.add_argument('--num_params', type=str, default='1.5B',
                    help='Model size parameter (e.g., 1.5B, 7B, 14B)')
parser.add_argument('--dataset_path', type=str, default='data/test/dataset.jsonl',
                    help='Path to dataset.jsonl file (default: data/test/dataset.jsonl)')
args = parser.parse_args()

# Dynamic directory names
model_base_name = f"qwen2.5-{args.num_params}-kg-finetuned"

# Step 1: Load the fine-tuned model and tokenizer
model_name = f"./{model_base_name}"  # Path to your locally fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="<pad>")  # Set a distinct pad_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Move the model to all available GPUs using DataParallel
# If CUDA is available and multiple GPUs are present, use them.
if torch.cuda.is_available():
    device = "cuda"  # Move model to GPU
    model = torch.nn.DataParallel(model)  # This will use all available GPUs
    model = model.cuda()  # Move the model to the GPUs
else:
    device = "cpu"  # Use CPU if no GPUs are available
    model = model.to(device)

# Ensure the model is in evaluation mode
model.eval()

# Step 3: Function to generate triples from the model's output
def generate_triples(text):
    # Pre-process the input text and tokenize
    inputs = tokenizer(f"Generate knowledge graph triples for this text: {text}", return_tensors="pt", padding=True, truncation=True)

    # Move input tensors to the same device as the model (GPUs in this case)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get the model's prediction
    with torch.no_grad():
        outputs = model.module.generate(  # Access the actual model wrapped in DataParallel
            inputs['input_ids'],
            max_new_tokens=512,  # Use max_new_tokens instead of max_length
            num_return_sequences=1,
            num_beams=1,  # Use greedy search, faster than beam search
            top_p=0.9,  # Top-p sampling to speed things up
            top_k=50,  # Limit sampling options for faster results
            early_stopping=False  # Disable early stopping
        )

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# Step 4: Load the test dataset
def load_test_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Step 5: Testing the model on the test set and saving results
def test_model_on_dataset(test_file_path):
    test_data = load_test_dataset(test_file_path)

    # Create a new file to save the generated triples
    output_file_path = os.path.join(
        os.path.dirname(test_file_path),  # Same directory as the test dataset
        f"generated_triples_finetuned_{args.num_params}.jsonl"
    )

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for idx, item in enumerate(test_data):
            print(f"Processing test sample {idx + 1}...")
            text = item['text']
            ground_truth_triples = item['triples']

            # Generate triples using the fine-tuned model
            generated_triples = generate_triples(text)

            # Save the results to the new JSONL file
            output_item = {
                "text": text,
                "ground_truth_triples": ground_truth_triples,
                "generated_triples": generated_triples
            }
            output_file.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    print(f"Generated triples saved to: {output_file_path}")

# Running the test
test_model_on_dataset(args.dataset_path)  # Use the dataset path from command-line arguments