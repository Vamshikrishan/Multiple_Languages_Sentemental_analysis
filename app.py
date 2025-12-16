import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Multilingual Sentiment Analyzer", layout="centered")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("final_model")
    model = AutoModelForSequenceClassification.from_pretrained("final_model")
    return tokenizer, model

tokenizer, model = load_model()
labels = ["positive", "negative", "neutral", "mixed"]

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def predict(text):
    inputs = tokenizer(
        clean_text(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=192
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    label_id = torch.argmax(probs).item()
    sentiment = labels[label_id]
    score = float(probs[label_id] * 2 - 1)

    if sentiment == "positive":
        reaction = "❤️ Love" if score > 0.8 else "👍 Like"
        emotion = "joy"
    elif sentiment == "negative":
        reaction = "😡 Angry" if score < -0.8 else "😢 Sad"
        emotion = "sadness"
    elif sentiment == "mixed":
        reaction = "😐 Mixed"
        emotion = "confusion"
    else:
        reaction = "😶 Neutral"
        emotion = "neutral"

    return sentiment, round(score, 2), emotion, reaction

st.title("🌍 Multilingual Sentiment Analyzer")
st.write("Supports **English, Telugu, Hindi, Urdu**")

user_input = st.text_area("Enter text:", height=120)

if st.button("Analyze"):
    if user_input.strip():
        sentiment, score, emotion, reaction = predict(user_input)

        st.success("Analysis Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Sentiment Score:** {score}")
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Reaction:** {reaction}")
    else:
        st.warning("Please enter some text.")
