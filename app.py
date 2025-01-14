import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Page Configuration
st.set_page_config(page_title="Sentence Similarity Checker", layout="centered")
st.title("Sentence Similarity Checker")

# Define layout: Columns for inputs and results
input_col, result_col = st.columns([1, 2])

# Input Section
with input_col:
    st.header("Input Sentences")
    source_sentence = st.text_input("Source Sentence:", "That is a happy person")
    st.markdown("**Sentences to Compare** (Enter one per line):")
    sentences_to_compare = st.text_area(
        "",
        value="That is a happy dog\nThat is a very happy person\nToday is a sunny day",
        height=150,
    ).splitlines()

# Compute similarity in real-time
if source_sentence and sentences_to_compare:
    source_embedding = model.encode(source_sentence, convert_to_tensor=True)
    comparison_embeddings = model.encode(sentences_to_compare, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(source_embedding, comparison_embeddings).squeeze(0)

# Result Section
with result_col:
    st.header("Results")
    if source_sentence and sentences_to_compare:
        for sentence, similarity in zip(sentences_to_compare, similarities):
            percentage = similarity.item() * 100
            st.markdown(f"**{sentence}**")
            st.progress(percentage / 100)  # Display progress bar
            st.write(f"Similarity: **{percentage:.2f}%**")
    else:
        st.info("Enter a source sentence and comparison sentences to view results.")
