import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies_df = pd.read_csv('movies.csv')

# Compute genres matrix
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
genres_matrix = vectorizer.fit_transform(movies_df['genres'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(genres_matrix, genres_matrix)
# Function to recommend movies by genres
def recommend_movies_by_genres(exclude_genres=[], include_genres=[], cosine_sim=cosine_sim, n=10):
    # Create a vector representing the genres of interest
    genres_vector = vectorizer.transform([''])  # Empty string as a placeholder

    # Calculate similarity scores between the specified genres and all movies
    sim_scores = cosine_similarity(genres_vector, genres_matrix).flatten()

    # Filter out movies with excluded genres
    for exclude_genre in exclude_genres:
        exclude_indices = [i for i, genre in enumerate(movies_df['genres']) if exclude_genre in genre]
        sim_scores[exclude_indices] = -1  # Set similarity score to -1 for excluded genres

    # Filter in movies with included genres
    for include_genre in include_genres:
        include_indices = [i for i, genre in enumerate(movies_df['genres']) if include_genre in genre]
        sim_scores[include_indices] *= 2  # Increase similarity score for included genres

    # Get the indices of top n similar movies
    movie_indices = sim_scores.argsort()[::-1][:n]

    # Return the top n recommended movies
    return movies_df.iloc[movie_indices]['title']

# Streamlit UI
st.title('Movie Recommendation System')

# Dropdown list for including genres
include_genres_input = st.multiselect('Exclude genres', options=sorted(vectorizer.vocabulary_.keys()))

# Dropdown list for excluding genres
exclude_genres_input = st.multiselect('Include genres', options=sorted(vectorizer.vocabulary_.keys()))


# Button to trigger recommendations
if st.button('Get Recommendations'):
    recommendations = recommend_movies_by_genres(include_genres=include_genres_input, exclude_genres=exclude_genres_input)
    st.write(recommendations)
