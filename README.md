# Clinical Note Generator

## Project Summary

The **Clinical Note Generator** is a tool that uses a fine-tuned sequence-to-sequence model to convert clinical dialogues into concise clinical notes. By leveraging advanced NLP techniques, this project aims to streamline the process of creating accurate, succinct summaries of patient interactions.

![How It Works](clinical-note-generator.gif)

## Data

We use the **ACI Bench** data, which is part of the [Clinical Visit Note Summarization Corpus](https://github.com/microsoft/clinical_visit_note_summarization_corpus?tab=readme-ov-file) provided by Microsoft. This dataset contains transcribed dialogues and corresponding reference notes, enabling the model to learn how to generate clinical notes effectively.

## Model

We fine-tuned a Hugging Face seq2seq model (by default, `facebook/bart-large-cnn`). However, you can experiment with other models (e.g., `google/flan-t5-base`) by updating the `--model_name` parameter in the fine-tuning script.

## Repository Structure

```plaintext
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── 1-data-exploration.ipynb
├── results/
│   └── metrics_facebook_bart-large-cnn.csv
├── scripts/
│   ├── evaluate_model.py
│   ├── finetune.py
│   └── prompt_engineering.py
├── summaries/
│   └── predictions_facebook_bart-large-cnn.csv
├── README.md
└── requirements.txt
```
- app/: Contains the Streamlit application for demo/interaction.
- notebooks/: Jupyter notebooks for data exploration and experimentation.
- results/: Stores evaluation metrics (e.g., ROUGE scores).
- scripts/: Python scripts for fine-tuning (finetune.py), evaluating (evaluate_model.py), and prompt engineering (prompt_engineering.py).
- summaries/: Contains generated summaries from the evaluation step.
- requirements.txt: Lists Python dependencies needed to run this project.

## Evaluation Results
| **rouge1** | **rouge2** | **rougeL** | **rougeLsum** |
|--------|--------|--------|-----------|
| 49.87  | 22.43  | 25.94  | 49.99     |

## Future Steps
1. **Address Hallucinations**: Further refine the model to avoid generating incorrect medical details (e.g., false symptoms, diseases, or medications).
2. **Explore Additional Architectures**: Experiment with other transformer-based models or emerging architectures to improve summarization quality.
3. **User Feedback Loop**: Integrate clinician feedback for continuous improvement and validation of generated notes.
4. **Enhanced Data Augmentation**: Incorporate more diverse clinical scenarios to make the model robust against varied dialogues.

##

**Note**: This project is intended for research and educational purposes. Always ensure that any automatically generated clinical content is reviewed and validated by qualified medical professionals.