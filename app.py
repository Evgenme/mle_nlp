import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example DataFrame
file_path = 'G:\Praktikum\MLE\Master_mle\data\mqp.csv'
data = pd.read_csv(file_path, names=['claster', 'lable', 'phrase', 'target'])

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Encode phrases to TF-IDF vectors
phrase_vectors = vectorizer.fit_transform(data['phrase'].tolist())

# Function to find the nearest neighbors
def find_nearest_neighbors(query, phrase_vectors, data, n_neighbors=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, phrase_vectors).flatten()
    nearest_indices = similarities.argsort()[-n_neighbors:][::-1]
    nearest_phrases = data['phrase'].iloc[nearest_indices].tolist()
    
    # Sort phrases by similarity from highest to lowest
    # sorted_phrases = sorted(zip(nearest_phrases, similarities[nearest_indices]), key=lambda x: x[1], reverse=True)
    # sorted_phrases = [phrase for phrase, similarity in sorted_phrases]
    #return sorted_phrases
    
    return nearest_phrases

# Streamlit app
st.title("Nearest Neighbor Phrases Finder")

# Input text
query = st.text_input("Enter your phrase:")

# Input number of neighbors
n_neighbors = st.number_input("Enter number of nearest neighbors:", min_value=1, max_value=10, value=5)

if st.button("Find Nearest Neighbors"):
    if query:
        nearest_neighbors = find_nearest_neighbors(query, phrase_vectors, data, n_neighbors)
        st.write("Nearest Neighbor Phrases:")
        for idx, phrase in enumerate(nearest_neighbors, start=1):
            st.write(f"{idx}. {phrase}")
    else:
        st.write("Please enter a phrase to search.")
