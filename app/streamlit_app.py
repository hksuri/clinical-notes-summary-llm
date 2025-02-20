import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set page config with clinic-themed colors
st.set_page_config(page_title="Clinical Note Summarizer", page_icon="💊", layout="centered")

# Load fine-tuned model
MODEL_NAME = "../../finetuned_models/facebook_bart-large-cnn"  # Change to your actual path
@st.cache_resource()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

# Load model
tokenizer, model = load_model()

# Apply blue and white clinic theme
st.markdown(
    """
    <style>
        body { background-color: #e0f7fa; color: #0277bd; }
        .stTextInput, .stButton { border-color: #0277bd; }
        h1 { color: #01579b; }
        .stAlert { background-color: #bbdefb; color: #01579b; }
        .ehr-note { font-family: 'Courier New', monospace; background-color: #f9f9f9; padding: 15px; border-left: 5px solid #0277bd; }
        .ehr-header { font-weight: bold; color: #01579b; font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    # Display an image related to clinical notes
    st.image("https://images.vexels.com/media/users/3/144224/isolated/preview/589394662ba164058d2ac84b4a0643b2-medical-record-table-notes.png", caption="Simplify your clinical documentation", width=150)

# Title
st.title("Clinical Note Summarizer")

# Text input
note_input = st.text_area("Paste your clinical note here:")

# Summarize button
if st.button("Summarize"):
    if note_input:
        with st.spinner("Summarizing..."):
            input_ids = tokenizer(note_input, return_tensors="pt", truncation=True, max_length=1024).input_ids
            output = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            st.subheader("Summary")
            st.write(summary)
    else:
        st.warning("Please paste a note to summarize.")

# Footer
st.markdown("---")
st.markdown("**Note:** This tool uses a fine-tuned Hugging Face model for dialogue-to-note summarization.")