"""
evaluate_model.py

Evaluate a fine-tuned seq2seq model on a test set.
Saves generated summaries to `summaries/` and metrics to `results/`.
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load as load_metric
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned seq2seq model on test data.")
    parser.add_argument("--model_path", type=str, default="../../finetuned_models/facebook_bart-large-cnn",
                        help="Path to the fine-tuned model directory, e.g. outputs/google_flan-t5-base.")
    parser.add_argument("--test_file", type=str, default="../../clinical_visit_note_summarization_corpus/data/aci-bench/challenge_data/clinicalnlp_taskB_test1.csv",
                        help="Path to the test CSV file.")
    
    parser.add_argument("--max_source_length", type=int, default=1024,
                        help="Max input length (dialogue).")
    parser.add_argument("--max_target_length", type=int, default=512,
                        help="Max length for generated summary.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference.")
    return parser.parse_args()

def main():
    args = parse_args()

    model_name = os.path.basename(args.model_path)  # e.g. "google_flan-t5-base"
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    print("Loading test dataset...")
    test_dataset = load_dataset("csv", data_files={"test": args.test_file})["test"]

    # Convert to list for easier iteration
    dialogues = test_dataset["dialogue"]
    references = test_dataset["note"]

    # Setup ROUGE
    rouge_metric = load_metric("rouge")

    # We'll generate predictions and store them
    predictions = []

    print("Generating predictions on the test set...")
    for i in range(0, len(dialogues), args.batch_size):
        batch_dialogues = dialogues[i : i + args.batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_dialogues,
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Generate
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            max_length=args.max_target_length,
            num_beams=4
        )
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)

    # Compute metrics
    # Format data for ROUGE (newline separated for each summary)
    pred_for_metric = ["\n".join(pred.strip().split()) for pred in predictions]
    ref_for_metric = ["\n".join(ref.strip().split()) for ref in references]

    result = rouge_metric.compute(predictions=pred_for_metric, references=ref_for_metric)
    rouge_scores = {
        key: (value.mid.fmeasure * 100 if hasattr(value, "mid") else value * 100)
        for key, value in result.items()
    }

    # Save predictions to a CSV in `summaries/`
    summary_dir = "../summaries"
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, f"predictions_{model_name}.csv")

    print(f"Saving generated summaries to {summary_file}")
    df_pred = pd.DataFrame({
        "dialogue": dialogues,
        "reference_note": references,
        "predicted_note": predictions
    })
    df_pred.to_csv(summary_file, index=False)

    # Save metrics to a CSV in `results/`
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"metrics_{model_name}.csv")

    print(f"Saving metrics to {results_file}")
    df_metrics = pd.DataFrame([rouge_scores])
    df_metrics.to_csv(results_file, index=False)

    print("Evaluation complete.")
    print("ROUGE Scores:", rouge_scores)


if __name__ == "__main__":
    main()