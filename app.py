# streamlit_app.py

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer from the correct folder
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    return model, tokenizer

model, tokenizer = load_model()

# App UI
st.title("📰 Fake News Detector")
st.write("Enter a news headline or article to check if it's **Fake** or **Real**.")

user_input = st.text_area("📝 Enter the news text:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()

        # Output result
        if prediction == 1:
            st.success(f"✅ This news is **Real** (Confidence: {confidence:.2f})")
        else:
            st.error(f"❌ This news is **Fake** (Confidence: {confidence:.2f})")
