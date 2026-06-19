import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

print("1. Loading cleaned movie database...")
df = pd.read_csv("movies_cleaned.csv")

# Fill missing values to avoid errors
df['metadata'] = df['metadata'].fillna('')

print("2. Training model (calculating TF-IDF matrix)...")
# Improved Vectorizer:
# - ngram_range=(1, 2): catches phrases like "time travel" or "sci fi"
# - min_df=2: ignores typos/tags that appear only once
# - sublinear_tf=True: prevents repeated tags from dominating the score
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
tfidf_matrix = tfidf.fit_transform(df['metadata'])

print("3. Saving trained weights...")
# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save tfidf matrix
with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

print("Done! vectorizer.pkl and tfidf_matrix.pkl have been created.")
