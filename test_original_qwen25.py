import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description='Test original Qwen2.5 model for triple generation')
parser.add_argument('--num_params', type=str, default='1.5B',
                    help='Model size parameter (e.g., 1.5B, 7B, 14B)')
parser.add_argument('--dataset_path', type=str, default='data/test/dataset.jsonl',
                    help='Path to dataset.jsonl file (default: data/test/dataset.jsonl)')
args = parser.parse_args()

# Step 1: Load the original model and tokenizer
model_name = f"Qwen/Qwen2.5-{args.num_params}-Instruct"  # Original model path
tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="<pad>")  # Set a distinct pad_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Move the model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Ensure the model is in evaluation mode
model.eval()

# Step 3: Define the system and user messages
system_message = {
    "role": "system",
    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your task is to generate knowledge graph triples from the given text. Each triple should be in the format: [\"subject\", \"predicate\", \"object\"]."
}

few_shot_examples = """
Example 1:
Text: Alaska is an instance of a United States Senate constituency. This means it is one of the designated regions represented by a senator in the U.S. Senate.
Triples:
["Alaska", "instance of", "United States Senate constituency"]

Example 2:
Text: Regina Ullmann was a Swiss poet and writer, born on December 14, 1884, in St. Gallen. She was a prominent figure in the field of literature, particularly known for her contributions to poetry. Regina, whose family name was Ullmann, was a female author who wrote and spoke German. Her work and legacy are documented in the SAPA Foundation, Swiss Archive of the Performing Arts. She was a member of both the German Academy for Language and Literature and the Bavarian Academy of Fine Arts, highlighting her significant influence in the literary world. Regina Ullmann passed away on January 6, 1961, in Ebersberg. Her life and works have been described in sources such as "Verbrannt, verboten, vergessen" and "Lexikon deutschsprachiger Schriftstellerinnen 1800–1945."
Triples:
["Regina Ullmann", "occupation", "poet"]
["Regina Ullmann", "documentation files at", "SAPA Foundation, Swiss Archive of the Performing Arts"]
["Regina Ullmann", "member of", "German Academy for Language and Literature"]
["Regina Ullmann", "family name", "Ullmann"]
["Regina Ullmann", "described by source", "Verbrannt, verboten, vergessen"]
["Regina Ullmann", "place of death", "Ebersberg"]
["Regina Ullmann", "given name", "Regina"]
["Regina Ullmann", "field of work", "literature"]
["Regina Ullmann", "languages spoken, written or signed", "German"]
["Regina Ullmann", "occupation", "writer"]
["Regina Ullmann", "place of birth", "St. Gallen"]
["Regina Ullmann", "field of work", "poetry"]
["Regina Ullmann", "country of citizenship", "Switzerland"]
["Regina Ullmann", "described by source", "Lexikon deutschsprachiger Schriftstellerinnen 1800–1945"]
["Regina Ullmann", "member of", "Bavarian Academy of Fine Arts"]
["Regina Ullmann", "date of death", "1961-01-06T00:00:00Z"]
["Regina Ullmann", "sex or gender", "female"]
["Regina Ullmann", "date of birth", "1884-12-14T00:00:00Z"]
["Regina Ullmann", "instance of", "human"]
"""

# Step 4: Function to generate triples from the model's output
def generate_triples(text):
    # Define the user message
    user_message = {
        "role": "user",
        "content": f"{few_shot_examples}\nNow generate triples for the following text:\nText: {text}\nTriples:\n"
    }

    # Combine the system and user messages
    messages = [system_message, user_message]

    # Convert messages to a prompt string
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get the model's prediction
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=2048,  # Generate up to 512 new tokens
            temperature=0.7,  # Set temperature
            top_p=0.8,  # Set top-p
            repetition_penalty=1.05  # Set repetition penalty
        )

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the triples (remove the prompt)
    triples = generated_text.split("Triples:\n")[-1].strip()
    return triples

# Step 5: Load the test dataset
def load_test_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Step 6: Testing the model on the test set and saving results
def test_model_on_dataset(test_file_path):
    test_data = load_test_dataset(test_file_path)

    # Create a new file to save the generated triples
    output_file_path = os.path.join(
        os.path.dirname(test_file_path),  # Same directory as the test dataset
        f"generated_triples_original_qwen2.5_{args.num_params}.jsonl"
    )

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for idx, item in enumerate(test_data):
            print(f"Processing test sample {idx + 1}...")
            text = item['text']
            ground_truth_triples = item['triples']

            # Generate triples using the original model
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