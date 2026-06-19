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
