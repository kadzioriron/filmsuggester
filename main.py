from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import io

app = FastAPI(title="Multi-Platform Movie Recommender API")

# --- ЗАГРУЗКА МОДЕЛЕЙ ПРИ СТАРТЕ ---
df = pd.read_csv("movies_cleaned.csv")
df['title_lower'] = df['title'].str.lower()

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# --- ПАРСЕРЫ ---

def get_letterboxd_watched(username):
    url = f"https://letterboxd.com/{username}/films/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200: return None
    soup = BeautifulSoup(response.text, "html.parser")
    return list(set([img["alt"] for img in soup.find_all("img") if img.get("alt")]))

def get_imdb_watched(user_id):
    # У IMDb ID выглядит как 'ur12345678'
    url = f"https://www.imdb.com/user/{user_id}/ratings"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200: return None
    soup = BeautifulSoup(response.text, "html.parser")
    # Ищем заголовки фильмов (у IMDb часто меняется верстка, это базовый вариант)
    titles = soup.find_all("h3", class_="lister-item-header")
    return list(set([t.find("a").text for t in titles if t.find("a")]))

# --- ЛОГИКА РЕКОМЕНДАЦИЙ (Вынесена отдельно, чтобы не дублировать код) ---
def generate_recommendations(user_films, username_or_source):
    if not user_films:
        raise HTTPException(status_code=400, detail="No movies found or profile is private")
        
    user_films_lower = [str(f).lower() for f in user_films]
    matched_indices = df[df['title_lower'].isin(user_films_lower)].index
    
    if len(matched_indices) == 0:
        return {"source": username_or_source, "status": "No matches in database", "recommendations": []}
        
    user_profile_vector = tfidf_matrix[matched_indices].mean(axis=0)
    sim_scores = cosine_similarity(np.asarray(user_profile_vector), tfidf_matrix).flatten()
    
    df['score'] = sim_scores
    recommendations = df[~df['title_lower'].isin(user_films_lower)].sort_values(by="score", ascending=False)
    
    results = [{"title": row['title'], "similarity_percent": round(row['score'] * 100, 1)} 
               for _, row in recommendations.head(10).iterrows()]
        
    return {
        "source": username_or_source,
        "movies_analyzed": len(matched_indices),
        "recommendations": results
    }

# --- ENDPOINTS (Точки доступа API) ---

# 1. Поиск по нику (Letterboxd или IMDb)
@app.get("/recommend/{platform}/{username}")
def get_recs_by_username(platform: str, username: str):
    platform = platform.lower()
    
    if platform == "letterboxd":
        user_films = get_letterboxd_watched(username)
    elif platform == "imdb":
        user_films = get_imdb_watched(username)
    else:
        raise HTTPException(status_code=400, detail="Unsupported platform. Use 'letterboxd' or 'imdb'")
        
    if user_films is None:
        raise HTTPException(status_code=404, detail="User not found or profile is hidden")
        
    return generate_recommendations(user_films, f"{platform}: {username}")

# 2. Универсальная загрузка CSV (Для Кинопоиска, Trakt и вообще чего угодно)
@app.post("/recommend/upload-csv")
async def get_recs_by_csv(file: UploadFile = File(...)):
    # Читаем загруженный файл в память
    contents = await file.read()
    try:
        # Пробуем прочитать как CSV
        uploaded_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file format")
    
    # Ищем колонку с названиями (разные сайты называют ее по-разному)
    possible_columns = ['Title', 'title', 'Name', 'name', 'Название', 'Movie', 'movie']
    title_col = next((col for col in possible_columns if col in uploaded_df.columns), None)
    
    if not title_col:
        raise HTTPException(status_code=400, detail=f"Could not find a movie title column. Looked for: {possible_columns}")
        
    user_films = uploaded_df[title_col].dropna().tolist()
    
    return generate_recommendations(user_films, f"file: {file.filename}")