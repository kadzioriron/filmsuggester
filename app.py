import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# 1. Letterboxd parser
def get_letterboxd_watched(username, max_pages=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    movies = []
    
    for page in range(1, max_pages + 1):
        url = f"https://letterboxd.com/{username}/films/page/{page}/"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            break
            
        soup = BeautifulSoup(response.text, "html.parser")
        imgs = soup.find_all("img")
        
        # If no movies found on this page, stop
        page_movies = [img.get("alt") for img in imgs if img.get("alt")]
        if not page_movies:
            break
            
        movies.extend(page_movies)
        
    return list(set(movies))

# 2. Load model and data (cached)
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("movies_cleaned.csv")
    # Remove year like (1995) for better matching with Letterboxd
    df['title_lower'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.lower()
    
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
        
    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_model_and_data()

# --- UI ---
st.title("🎬 Movie Recommender")
username = st.text_input("Letterboxd Username (e.g., username):")

if st.button("Find Recommendations"):
    if username:
        with st.spinner("Analyzing profile..."):
            user_films = get_letterboxd_watched(username)
            
            if user_films:
                user_films_lower = [f.lower() for f in user_films]
                
                # Find viewed movies in our db
                matched_indices = df[df['title_lower'].isin(user_films_lower)].index
                
                if len(matched_indices) == 0:
                    st.warning("None of your watched movies are in our top-15000 database :(")
                else:
                    st.success(f"Found {len(matched_indices)} of your movies in the database! Generating top...")
                    
                    # 3. Prediction logic
                    # Average vectors of watched movies to get taste profile
                    user_profile_vector = tfidf_matrix[matched_indices].mean(axis=0)
                    
                    # Calculate cosine similarity with all movies
                    sim_scores = cosine_similarity(np.asarray(user_profile_vector), tfidf_matrix).flatten()
                    
                    df['score'] = sim_scores
                    
                    # Filter out already watched and sort
                    recommendations = df[~df['title_lower'].isin(user_films_lower)].sort_values(by="score", ascending=False)
                    
                    st.subheader("🔥 Recommended for you:")
                    for idx, row in recommendations.head(10).iterrows():
                        st.write(f"**{row['title']}** (Similarity: {round(row['score'] * 100, 1)}%)")
            else:
                st.error("User not found or diary is hidden.")
    else:
        st.error("Please enter a username!")
