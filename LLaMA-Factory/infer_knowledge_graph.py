import json
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Infer knowledge graphs from text using Qwen model")
    parser.add_argument("--num_params", type=str, default="1.5B", 
                        help="Model size (e.g., '1.5B', '7B', '14B', '72B')")
    parser.add_argument("--use_finetuned", action="store_true", 
                        help="Use fine-tuned model instead of original Qwen model")
    args = parser.parse_args()
    
    # Determine model path
    if args.use_finetuned:
        model_path = f"saves/qwen2.5-{args.num_params}/full/sft"
        print(f"Using fine-tuned model from {model_path}")
    else:
        model_path = f"Qwen/Qwen2.5-{args.num_params}-Instruct"
        print(f"Using original Qwen model: {model_path}")
    
    # Load the model and tokenizer
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load test data
    test_file = "data/T2G_test.json"
    print(f"Loading test data from {test_file}")
    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Generate predictions
    results = []
    for i, item in enumerate(tqdm(test_data, desc="Processing examples")):
        # Format messages using the Qwen chat template
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{item['instruction']}\n{item['input']}"}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        
        # Extract only the generated part (excluding the input)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        results.append({
            "text": item["input"],
            "prediction": response,
            "ground_truth": item["output"]
        })
    
    # Save results
    model_type = "finetuned" if args.use_finetuned else "original"
    output_file = f"results/test_predictions_{model_type}_{args.num_params}.json"
    print(f"Saving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Completed! Processed {len(test_data)} examples.")

if __name__ == "__main__":
    main() 