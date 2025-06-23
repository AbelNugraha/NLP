
import streamlit as st
import torch
from transformers import BertTokenizer
from model.emotion_model import EmotionClassifier, predict_emotion

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("💬 Deteksi Emosi dari Teks")
st.markdown("Masukkan kalimat berbahasa Inggris dan dapatkan emosi yang terdeteksi.")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionClassifier(n_classes=6)
    model.load_state_dict(torch.load("model/bert_emotion_model.pt", map_location=device))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer, device

model, tokenizer, device = load_model()

text = st.text_area("Masukkan kalimat...", height=150)
if st.button("Deteksi Emosi"):
    if text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        prediction = predict_emotion(text, model, tokenizer, device)
        st.success(f"Emosi yang terdeteksi: **{prediction.upper()}**")
