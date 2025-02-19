"""
finetune.py

Script to fine-tune a Hugging Face seq2seq model (e.g., T5, Flan-T5, BART, Pegasus)
on a dataset of dialogues -> summarized notes.

Usage:
  python finetune.py --model_name google/flan-t5-base --num_train_epochs 3
"""

import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from evaluate import load as load_metric
import torch
torch.mps.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a seq2seq model for dialogue-to-note summarization.")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                        help="Hugging Face model identifier, e.g. google/flan-t5-base")
    parser.add_argument("--train_file", type=str, default="../../clinical_visit_note_summarization_corpus/data/aci-bench/challenge_data/train.csv",
                        help="Path to the training CSV file.")
    parser.add_argument("--val_file", type=str, default="../../clinical_visit_note_summarization_corpus/data/aci-bench/challenge_data/valid.csv",
                        help="Path to the validation CSV file.")
    
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Initial learning rate.")
    
    parser.add_argument("--max_source_length", type=int, default=3050,
                        help="Max input sequence length (dialogue).")
    parser.add_argument("--max_target_length", type=int, default=900,
                        help="Max target sequence length (note).")
    
    parser.add_argument("--output_dir", type=str, default="../../finetuned_models",
                        help="Directory to store fine-tuned model.")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Load dataset
    print("Loading dataset...")
    data_files = {
        "train": args.train_file,
        "validation": args.val_file
    }
    raw_datasets = load_dataset("csv", data_files=data_files)

    # Preprocess function
    def preprocess_function(examples):
        inputs = [dialogue for dialogue in examples["dialogue"]]
        targets = [note for note in examples["note"]]
        
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=args.max_target_length,
                padding="max_length",
                truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # Define training arguments
    model_output_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_"))
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir="../../logs",
        predict_with_generate=True,
        fp16=True,
        # possible push_to_hub or other arguments
    )

    # Load evaluation metric (e.g., ROUGE)
    rouge_metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE expects newline separated sentences
        decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
        decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]

        result = rouge_metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract a few ROUGE scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return result

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {model_output_dir}")
    trainer.save_ (model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    # # Print final metrics
    # print("Final metrics:", trainer.evaluate(eval_dataset=tokenized_datasets["validation"]))

    print("Fine-tuning completed.")


if __name__ == "__main__":
    main()

