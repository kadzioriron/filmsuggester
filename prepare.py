import pandas as pd

print("1. Reading files (this may take a minute)...")
movies = pd.read_csv("ml-25m/movies.csv")
ratings = pd.read_csv("ml-25m/ratings.csv")
tags = pd.read_csv("ml-25m/tags.csv")

print("2. Finding top-15000 most popular movies...")
# Count ratings per movie, get top 15000
popular_movies = ratings['movieId'].value_counts().head(15000).index
movies = movies[movies['movieId'].isin(popular_movies)]

print("3. Collecting user tags...")
tags['tag'] = tags['tag'].fillna('').astype(str)
# Group all tags for a movie into a single string
tags_grouped = tags[tags['movieId'].isin(popular_movies)].groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()

print("4. Merging data...")
df = pd.merge(movies, tags_grouped, on='movieId', how='left')
df['tag'] = df['tag'].fillna('')

# Convert "Action|Sci-Fi" to "Action Sci-Fi"
df['genres'] = df['genres'].str.replace('|', ' ', regex=False)

# Final metadata column: genres + tags
df['metadata'] = df['genres'] + " " + df['tag']

# Keep only necessary columns
ready_df = df[['title', 'metadata']]

# Save the lightweight file
ready_df.to_csv("movies_cleaned.csv", index=False)
print("Success! movies_cleaned.csv is ready.")
