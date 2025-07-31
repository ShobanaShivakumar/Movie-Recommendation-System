import pandas as pd
import numpy as np
import nltk
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import streamlit as st

# Reading the CSV
def load_data(csv_file="C:/Users/SHOBANA/Downloads/IMDB-Movie-Recommendation-System-Using-Storylines-main/IMDB-Movie-Recommendation-System-Using-Storylines-main/imdb_movies_2024.csv"):
    df = pd.read_csv(csv_file)
    # Dropping rows where Movie Name or Storyline are missing
    df.dropna(subset=["Title", "Description"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    text = " ".join(tokens)
    return text

# TF-IDF Matrix
def build_tfidf_matrix(df):
    df["cleaned_storyline"] = df["Description"].apply(clean_text)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["cleaned_storyline"])
    return tfidf, tfidf_matrix

# Cosine Similarity
def get_cosine_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend Movies
def recommend_movies(input_storyline, df, tfidf, tfidf_matrix, top_n=5):
    cleaned_input = clean_text(input_storyline)
    input_vector = tfidf.transform([cleaned_input])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    results = df.iloc[top_indices][["Title", "Description"]]
    results["Similarity Score"] = similarity_scores[top_indices]
    return results

# Load and Prepare
@st.cache_data
def load_and_prepare_data():
    df = load_data()
    tfidf, tfidf_matrix = build_tfidf_matrix(df)
    return df, tfidf, tfidf_matrix

# Main App
def main():
    st.title("IMDb Movie Recommendation System (2024)")

    df, tfidf, tfidf_matrix = load_and_prepare_data()

    st.write("""
    **Instructions**:  
    1. Enter a brief storyline or plot description in the text box below.  
    2. Click 'Recommend Movies' to see the top 5 similar movies.  
    """)

    user_input = st.text_area("Enter a movie storyline/plot here...")

    if st.button("Recommend Movies"):
        if user_input.strip() == "":
            st.warning("Please enter a storyline first.")
        else:
            recommendations = recommend_movies(user_input, df, tfidf, tfidf_matrix, top_n=5)
            st.subheader("Top 5 Recommended Movies:")
            for i, row in recommendations.iterrows():
                st.markdown(f"**{row['Title']}**")
                st.write(f"Similarity Score: {row['Similarity Score']:.4f}")
                st.write(f"Storyline: {row['Description']}")
                st.write("---")

if __name__ == "__main__":
    main()
