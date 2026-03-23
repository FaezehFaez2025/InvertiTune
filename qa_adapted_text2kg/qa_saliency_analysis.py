import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import defaultdict
import nltk
import argparse
import sys

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize

class GenQASaliencyAnalyzer:
    def __init__(self, model_name="allenai/unifiedqa-t5-base"):
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
    
    def generate_answer(self, context, question, max_length=64):
        inputs = self.tokenizer(
            question + " </s> " + context,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        generated_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_ids, answer
    
    def compute_saliency(self, context, question, generated_ids):
        # Re-encode to get input_ids, attention_mask, and correct offset_mapping
        encoding = self.tokenizer.encode_plus(
            question, context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        )
        input_ids      = encoding["input_ids"]          # [1, seq_len]
        attention_mask = encoding["attention_mask"]     # [1, seq_len]
        offsets        = encoding["offset_mapping"][0]  # [(start,end), ...] for seq_len tokens

        # Embed + grad
        embed_layer    = self.model.get_input_embeddings()
        inputs_embeds  = embed_layer(input_ids)         # [1, seq_len, hid_dim]
        inputs_embeds.requires_grad_(True)
        inputs_embeds.retain_grad()
        self.model.zero_grad()

        # Forward with labels to compute loss
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=generated_ids
        )
        loss = outputs.loss
        loss.backward()

        # Get per-token saliency
        grads           = inputs_embeds.grad.squeeze(0)    # [seq_len, hid_dim]
        token_importance= grads.norm(dim=-1)              # [seq_len]
        if token_importance.numel() > 0:
            token_importance = token_importance / token_importance.norm()

        return token_importance.detach(), offsets
    
    def analyze_and_print(self, context, question, top_k=3):
        gen_ids, answer = self.generate_answer(context, question)
        token_imp, offsets = self.compute_saliency(context, question, gen_ids)

        # Map token saliency onto character positions
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(question, context, truncation=True, max_length=512)
        )
        char_scores = defaultdict(float)
        for (start, end), score in zip(offsets, token_imp):
            if start == end:
                continue
            for i in range(start, end):
                char_scores[i] += score.item()

        # Score sentences
        sentences = sent_tokenize(context)
        sent_scores = []
        for sent in sentences:
            idx = context.find(sent)
            if idx >= 0:
                sc = sum(char_scores.get(i, 0.0) for i in range(idx, idx + len(sent)))
                sent_scores.append((sent.strip(), sc))

        ranked = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:top_k]

        # Print everything
        print(f"Question : {question}")
        print(f"Context  : {context[:100]}{'...' if len(context)>100 else ''}\n")
        print("Answer   :")
        print("-"*50)
        print(answer)
        print("-"*50 + "\n")
        print(f"Top {top_k} relevant sentences:")
        print("-"*50)
        for i,(s,sc) in enumerate(ranked,1):
            print(f"{i}. {s} (score: {sc:.4f})")
        return answer, ranked

def get_user_input():
    print("=== Generative QA + Saliency ===")
    print("Enter context (press Enter twice when done):")
    lines=[]
    while True:
        line = input()
        if not line.strip() and lines:
            break
        if line.strip():
            lines.append(line)
    context = " ".join(lines).strip()
    if not context:
        print("Error: No context provided!"); return None, None
    question = input("\nEnter question: ").strip()
    if not question:
        print("Error: No question provided!"); return None, None
    return context, question

def main():
    parser = argparse.ArgumentParser(description='Generative QA + Saliency Tool')
    parser.add_argument('--text','-t',     type=str, help='Context text')
    parser.add_argument('--question','-q', type=str, help='Question')
    parser.add_argument('--model','-m',    type=str, default="allenai/unifiedqa-t5-base",
                        help='Seq2seq QA model name')
    parser.add_argument('--top_k','-k',    type=int, default=3, help='Number of sentences')
    parser.add_argument('--interactive','-i', action='store_true')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    analyzer = GenQASaliencyAnalyzer(model_name=args.model)

    if args.interactive or not (args.text and args.question):
        context, question = get_user_input()
        if not context:
            sys.exit(1)
    else:
        context, question = args.text, args.question

    print("\n" + "="*60)
    analyzer.analyze_and_print(context, question, top_k=args.top_k)
    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
