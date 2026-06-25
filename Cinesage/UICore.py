import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Configuration
# -----------------------------
load_dotenv()

st.set_page_config(page_title="Information Extractor", page_icon="📄", layout="wide")

# -----------------------------
# LLM
# -----------------------------
model = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Information Extraction Assistant.

Your task is to analyze the provided text and extract the most useful and important information in a clear, well-organized format.

Instructions:

1. Identify the main subject of the text.
2. Extract all important entities, facts, and relationships.
3. Organize information into relevant categories.
4. Include only information explicitly stated or strongly supported by the text.
5. Avoid repeating information.
6. Generate a concise summary at the end.
7. If a category is not applicable, omit it.

Extract information using the following structure:

# Title / Main Subject

## Overview
- Brief description of the subject.

## Key Details
- Important facts and information.

## People Mentioned
- Person name
- Role / Contribution

## Organizations Mentioned
- Organization name
- Relevance

## Locations Mentioned
- Location name
- Relevance

## Dates / Time Information
- Any dates, years, or periods mentioned.

## Technical / Scientific Concepts
- Concepts and explanations.

## Themes / Topics
- Main themes discussed.

## Keywords
- Important keywords for search and indexing.

## Quick Summary
- 2-3 sentence summary capturing the core message.
""",
        ),
        (
            "human",
            """
Analyze the following text and extract useful information:

{text}
""",
        ),
    ]
)

# -----------------------------
# Header
# -----------------------------
st.title("📄 Information Extraction Assistant")

st.markdown(
    """
Paste any paragraph, article, document section, or text snippet below.
The AI will analyze it and extract the most useful information in a structured format.
"""
)

# -----------------------------
# Input
# -----------------------------
user_text = st.text_area(
    "Enter Text", height=300, placeholder="Paste your paragraph here..."
)

# -----------------------------
# Button
# -----------------------------
if st.button("🚀 Extract Information", use_container_width=True):

    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing text..."):

            final_prompt = prompt.invoke({"text": user_text})

            response = model.invoke(final_prompt)

        st.divider()

        st.subheader("📌 Extracted Information")

        st.markdown(response.content)
