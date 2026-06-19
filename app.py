import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# 1. Letterboxd parser
def get_letterboxd_watched(username):
    url = f"https://letterboxd.com/{username}/films/"
    
    # Маскируем скрипт под настоящий Google Chrome на Windows
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://google.com/"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # Если сайт всё равно заблокировал запрос, выводим реальный код ошибки на экран
        if response.status_code != 200:
            st.error(f"Ошибка доступа к Letterboxd. Код ответа сервера: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Собираем названия фильмов
        films = [img["alt"] for img in soup.find_all("img") if img.get("alt")]
        
        # Letterboxd иногда подтягивает аватарку юзера в img alt, отфильтруем пустые значения
        return list(set([f for f in films if f.strip()]))
        
    except Exception as e:
        st.error(f"Ошибка соединения: {e}")
        return []

from huggingface_hub import hf_hub_download
import os

# 2. Load model and data (cached)
@st.cache_resource
def load_model_and_data():
    # Название репозитория, куда вы загрузили файлы (замените если отличается)
    repo_id = "kadzioriron/filmsuggester" 
    
    # Скачиваем файлы из HF Hub
    df_path = hf_hub_download(repo_id=repo_id, filename="movies_cleaned.csv")
    vectorizer_path = hf_hub_download(repo_id=repo_id, filename="vectorizer.pkl")
    tfidf_path = hf_hub_download(repo_id=repo_id, filename="tfidf_matrix.pkl")

    df = pd.read_csv(df_path)
    # Remove year like (1995) for better matching with Letterboxd
    df['title_lower'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.lower()
    
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        tfidf_matrix = pickle.load(f)
        
    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_model_and_data()

# --- UI ---
st.title("🎬 AI Movie Recommender")

st.markdown("""
### Welcome to the AI Movie Recommender! 🍿
This application uses a Machine Learning algorithm called **TF-IDF** (Term Frequency-Inverse Document Frequency) and **Cosine Similarity** to analyze the metadata of over 15,000 popular movies. 

**How it works:**
1. It analyzes the genres and thousands of user-generated tags for each movie.
2. It builds a mathematical "taste profile" based on the movies you like.
3. It compares your taste profile with the entire database to find the highest mathematical match.

You can either parse your public **Letterboxd** profile, or **manually select** your favorite movies and genres!
---
""")

tab1, tab2 = st.tabs(["📊 Letterboxd Profile", "🎯 Manual Selection"])

with tab1:
    st.subheader("Import from Letterboxd")
    username = st.text_input("Letterboxd Username (e.g., kadzioriron):")

    if st.button("Find Recommendations from Profile"):
        if username:
            with st.spinner("Analyzing profile and parsing pages..."):
                user_films = get_letterboxd_watched(username)
                
                if user_films:
                    user_films_lower = [f.lower() for f in user_films]
                    
                    matched_indices = df[df['title_lower'].isin(user_films_lower)].index
                    
                    if len(matched_indices) == 0:
                        st.warning("None of your watched movies are in our top-15000 database :(")
                    else:
                        st.success(f"Successfully parsed {len(user_films)} movies! Found {len(matched_indices)} of them in our database. Generating top...")
                        
                        user_profile_vector = tfidf_matrix[matched_indices].mean(axis=0)
                        sim_scores = cosine_similarity(np.asarray(user_profile_vector), tfidf_matrix).flatten()
                        
                        df['score'] = sim_scores
                        recommendations = df[~df['title_lower'].isin(user_films_lower)].sort_values(by="score", ascending=False)
                        
                        st.subheader("🔥 Recommended for you:")
                        for idx, row in recommendations.head(15).iterrows():
                            st.write(f"**{row['title']}** (Similarity: {round(row['score'] * 100, 1)}%)")
                else:
                    st.error("User not found, diary is hidden, or there are no logged movies.")
        else:
            st.error("Please enter a username!")

with tab2:
    st.subheader("Build your custom taste profile")
    
    # Selection of favorite movies from the database
    selected_movies = st.multiselect(
        "Select some of your favorite movies:",
        options=df['title'].tolist(),
        max_selections=5,
        help="Type to search through 15,000 movies"
    )
    
    # Generic popular genres for manual input
    popular_genres = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    selected_genres = st.multiselect(
        "Select your favorite genres:",
        options=popular_genres
    )
    
    if st.button("Get Custom Recommendations"):
        if not selected_movies and not selected_genres:
            st.warning("Please select at least one movie or genre!")
        else:
            with st.spinner("Crunching the numbers..."):
                vectors_to_average = []
                
                # Add vectors for selected movies
                if selected_movies:
                    movie_indices = df[df['title'].isin(selected_movies)].index
                    movie_vectors = tfidf_matrix[movie_indices]
                    # Convert to dense array to avoid sparse matrix shape issues
                    vectors_to_average.append(np.asarray(movie_vectors.mean(axis=0)))
                
                # Add pseudo-vector for selected genres
                if selected_genres:
                    genres_string = " ".join(selected_genres).lower()
                    genre_vector = vectorizer.transform([genres_string])
                    # Convert to dense array
                    vectors_to_average.append(genre_vector.toarray())
                
                # Combine movie vectors and genre vectors
                if len(vectors_to_average) == 2:
                    # Give slightly more weight to actual movies than just generic genres
                    final_profile = (vectors_to_average[0] * 0.7) + (vectors_to_average[1] * 0.3)
                else:
                    final_profile = vectors_to_average[0]
                
                # Ensure the final profile is a 2D array for cosine_similarity
                if len(final_profile.shape) == 1:
                    final_profile = final_profile.reshape(1, -1)
                elif len(final_profile.shape) == 3:
                     # Just in case we ended up with an extra dimension
                     final_profile = final_profile.reshape(1, -1)
                
                sim_scores = cosine_similarity(final_profile, tfidf_matrix).flatten()
                df['score'] = sim_scores
                
                # Filter out the movies the user selected as input
                recommendations = df[~df['title'].isin(selected_movies)].sort_values(by="score", ascending=False)
                
                st.subheader("✨ Your Custom Recommendations:")
                for idx, row in recommendations.head(15).iterrows():
                    st.write(f"**{row['title']}** (Match Score: {round(row['score'] * 100, 1)}%)")
