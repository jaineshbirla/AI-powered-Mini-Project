"""
AI-Powered Text Summarizer & Sentiment Analyzer
-----------------------------------------------
An interactive web application built with Streamlit and Hugging Face 
transformers to analyze text sentiment and generate summaries.
"""

import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="AI Text Analyzer & Summarizer",
    page_icon="🤖",
    layout="centered"
)

# Load AI Pipelines (Cached for performance)
@st.cache_resource
def load_ai_models():
    # Using lightweight, fast Hugging Face open-source models
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return summarizer, sentiment_analyzer

with st.spinner("Loading AI models into memory... Please wait."):
    summarizer, sentiment_analyzer = load_ai_models()

# UI Header
st.title("🤖 AI-Powered Text Analyzer & Summarizer")
st.write("Leverage state-of-the-art Natural Language Processing (NLP) models to instantly analyze sentiment and summarize long blocks of text.")

# Text Input Area
user_text = st.text_area("Enter or paste your text below:", height=200, placeholder="Type an article, review, or paragraph here...")

if st.button("Run AI Analysis", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text before running the analysis.")
    else:
        with st.spinner("Analyzing text with AI..."):
            
            # 1. Sentiment Analysis
            sentiment_result = sentiment_analyzer(user_text)[0]
            sentiment_label = sentiment_result['label']
            sentiment_score = sentiment_result['score'] * 100

            # 2. Text Summarization (if text is long enough)
            word_count = len(user_text.split())
            if word_count > 30:
                # Adjust max_length safely based on input length
                max_len = min(max(30, word_count // 2), 130)
                min_len = max(10, max_len // 3)
                summary_result = summarizer(user_text, max_length=max_len, min_length=min_len, do_sample=False)
                summary_text = summary_result[0]['summary_text']
            else:
                summary_text = "Text is too short for summarization (needs more than 30 words). Here is your original text: " + user_text

        # Display Results
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Analysis")
            if sentiment_label == "POSITIVE":
                st.markdown(f"### 🟢 **Positive** ({sentiment_score:.1f}% confidence)")
            else:
                st.markdown(f"### 🔴 **Negative** ({sentiment_score:.1f}% confidence)")

        with col2:
            st.subheader("📉 Text Statistics")
            st.write(f"- **Word Count:** {word_count}")
            st.write(f"- **Character Count:** {len(user_text)}")

        st.markdown("---")
        st.subheader("📝 AI Generated Summary")
        st.info(summary_text)

# Footer
st.markdown("---")
st.caption("Built with Python, Streamlit, and Hugging Face Transformers as part of AI/ML Learning Journey.")
