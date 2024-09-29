import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the fine-tuned model and tokenizer (adjust paths as necessary)
@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained('./trained_model')  # Replace with your model path
    tokenizer = AutoTokenizer.from_pretrained('./trained_model')
    return model, tokenizer

# Function to generate summaries
def generate_summary(article_text, model, tokenizer):
    inputs = tokenizer(article_text, max_length=1024, padding='max_length', truncation=True, return_tensors="pt")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    
    # Generate the summary
    with torch.no_grad():
        summary_ids = model.generate(inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
def main():
    st.title("Text Summarization Web App")
    st.write("Enter an article and get a summarized version using the fine-tuned model.")

    # Input article from the user
    article_text = st.text_area("Enter Article Text", height=200)
    
    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Button to summarize
    if st.button("Summarize"):
        if article_text.strip() == "":
            st.warning("Please enter an article text!")
        else:
            st.write("Generating summary...")
            summary = generate_summary(article_text, model, tokenizer)
            st.subheader("Generated Summary")
            st.write(summary)

if __name__ == "__main__":
    main()
