import json
import torch
import argparse
import os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, TrainerCallback
)
from datasets import Dataset

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Argument parsing
parser = argparse.ArgumentParser(description='Fine-tune Qwen2.5 model')
parser.add_argument('--num_params', type=str, default='1.5B',
                    help='Model size parameter (e.g., 1.5B, 7B, 14B)')
parser.add_argument('--dataset_path', type=str, default='data/train/dataset.jsonl',
                    help='Path to dataset.jsonl file (default: data/train/dataset.jsonl)')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of training epochs (default: 10)')
args = parser.parse_args()

# Dynamic directory names
model_base_name = f"qwen2.5-{args.num_params}-kg-finetuned"

# Step 1: Dataset loading
def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            triples_str = "\n".join([f"- {t[0]} → {t[1]} → {t[2]}" for t in item["triples"]])
            data.append({
                "text": item["text"],
                "triples": triples_str
            })
    
    dataset = Dataset.from_dict({
        "text": [item["text"] for item in data],
        "triples": [item["triples"] for item in data]
    })
    print(f"\nDataset loaded successfully:")
    print(f"Total number of examples: {len(dataset)}")
    print(f"Number of training steps per epoch: {len(dataset) // (4 * 4)}")  # batch_size * gradient_accumulation
    return dataset

# Step 2: Model and tokenizer initialization
model_name = f"Qwen/Qwen2.5-{args.num_params}-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="right",
    pad_token="<|endoftext|>",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# Step 3: Tokenization function
def tokenize_function(examples):
    conversations = []
    for text, triples in zip(examples["text"], examples["triples"]):
        conversations.append([
            {"role": "user", "content": f"Generate knowledge graph triples for this text: {text}"},
            {"role": "assistant", "content": triples}
        ])
    
    user_texts = [conv[0]['content'] for conv in conversations]
    assistant_texts = [conv[1]['content'] for conv in conversations]
    
    user_tokenized = tokenizer(
        user_texts,  
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
        add_special_tokens=True
    )

    assistant_tokenized = tokenizer(
        assistant_texts,  
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
        add_special_tokens=True
    )

    input_ids = torch.cat([user_tokenized['input_ids'], assistant_tokenized['input_ids']], dim=1)
    attention_mask = torch.cat([user_tokenized['attention_mask'], assistant_tokenized['attention_mask']], dim=1)
    labels = input_ids.clone()

    assistant_token = tokenizer.encode("assistant", add_special_tokens=False)
    
    for conv_idx in range(len(conversations)):
        user_tokens = user_tokenized['input_ids'][conv_idx]
        assistant_tokens = assistant_tokenized['input_ids'][conv_idx]  # Assistant tokens are after user tokens
        
        assistant_start = len(user_tokens)
        labels[conv_idx, :assistant_start] = -100  # Mask the user part in the labels
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Step 4: Callback
class PrintEpochProgressCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch}/{args.num_train_epochs} starting...")

# Step 5: Dataset preparation
dataset = load_dataset(args.dataset_path)
tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=32)

# Step 6: Training arguments
training_args = TrainingArguments(
    output_dir=f"./{model_base_name}",
    learning_rate=2e-5,  # Adjusted for instruction tuning
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=args.num_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
    bf16=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    optim="adamw_torch",
    max_grad_norm=0.3,
    dataloader_num_workers=4,  # Added for better data loading
    dataloader_pin_memory=True  # Added for better performance
)

# Step 7: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    callbacks=[PrintEpochProgressCallback()]  # Add the custom callback
)

# Start training
trainer.train()

# Step 8: Save final model
model.save_pretrained(f"./{model_base_name}")
tokenizer.save_pretrained(f"./{model_base_name}")