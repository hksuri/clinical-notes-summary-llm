"""
prompt_engineering.py

Perform prompt-based inference (no fine-tuning) to convert dialogues into notes
using a Hugging Face model, saving the outputs to `summaries/` and metrics to `results/`.

Usage:
  python prompt_engineering.py --model_name google/flan-t5-large
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from evaluate import load as load_metric

def parse_args():
    parser = argparse.ArgumentParser(description="Prompt-based inference for dialogue->note summarization.")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                        help="Hugging Face model ID for inference.")
    parser.add_argument("--test_file", type=str, default="data/test.csv",
                        help="Path to test CSV file.")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    model_name_clean = args.model_name.replace("/", "_")

    print(f"Loading model for prompt engineering: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    print("Loading test data...")
    test_dataset = load_dataset("csv", data_files={"test": args.test_file})["test"]
    dialogues = test_dataset["dialogue"]
    references = test_dataset["note"]

    # A very basic prompt template
    prompt_template = (
        "You are a helpful medical assistant. Transform the following conversation "
        "into a concise clinical note capturing key details:\n\n"
        "CONVERSATION:\n{dialogue}\n\n"
        "NOTE:"
    )

    rouge_metric = load_metric("rouge")
    predictions = []

    print("Running prompt-based inference...")
    for i in range(0, len(dialogues), args.batch_size):
        batch_dialogues = dialogues[i : i + args.batch_size]
        batch_prompts = [prompt_template.format(dialogue=dlg) for dlg in batch_dialogues]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with model.cuda() if model.device.type == "cuda" else model.cpu():
            outputs = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_length=args.max_target_length,
                num_beams=4,
                temperature=0.7
            )

        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)

    # Compute metrics
    pred_for_metric = ["\n".join(pred.strip().split()) for pred in predictions]
    ref_for_metric = ["\n".join(ref.strip().split()) for ref in references]
    result = rouge_metric.compute(predictions=pred_for_metric, references=ref_for_metric)
    rouge_scores = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Save predictions
    os.makedirs("../clinical-notes-summary-llm/summaries", exist_ok=True)
    summary_file = f"summaries/prompt_predictions_{model_name_clean}.csv"
    df_pred = pd.DataFrame({
        "dialogue": dialogues,
        "reference_note": references,
        "prompt_generated_note": predictions
    })
    df_pred.to_csv(summary_file, index=False)
    print(f"Saved prompt-based summaries to {summary_file}")

    # Save metrics
    os.makedirs("../clinical-notes-summary-llm/results", exist_ok=True)
    results_file = f"results/prompt_metrics_{model_name_clean}.csv"
    df_metrics = pd.DataFrame([rouge_scores])
    df_metrics.to_csv(results_file, index=False)
    print(f"Saved prompt-based metrics to {results_file}")

    print("Prompt-based inference completed.")
    print("ROUGE Scores:", rouge_scores)

if __name__ == "__main__":
    main()