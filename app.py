import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics import precision_recall_fscore_support
import streamlit as st

movies = pd.read_csv('movie.csv')
ratings = pd.read_csv('rating.csv')
tags = pd.read_csv('tag.csv')
genome_scores = pd.read_csv('genome_scores.csv')
genome_tags = pd.read_csv('genome_tags.csv')

movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
movie_tag_matrix = genome_scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)

knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(movie_tag_matrix)

def get_movie_id(movie_name):
    movie = movies[movies["title"].str.contains(movie_name, case=False, na=False)]
    return movie.iloc[0]["movieId"] if not movie.empty else None

def get_similar_movies(movie_id, k=5):
    if movie_id not in movie_tag_matrix.index:
        return "Movie ID not found"
    distances, indices = knn_model.kneighbors([movie_tag_matrix.loc[movie_id]], n_neighbors=k+1)
    similar_movie_ids = [movie_tag_matrix.index[i] for i in indices.flatten()[1:]]
    return movies[movies["movieId"].isin(similar_movie_ids)][["movieId", "title"]]

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2)

svd_model = SVD()
svd_model.fit(trainset)

def predict_rating(user_id, movie_id):
    return svd_model.predict(user_id, movie_id).est

def hybrid_recommend(user_id, movie_id, k=5, alpha=0.7):
    svd_score = predict_rating(user_id, movie_id)
    similar_movies_df = get_similar_movies(movie_id, k)

    hybrid_scores = []
    for m_id in similar_movies_df["movieId"]:
        sim_movie_svd_score = predict_rating(user_id, m_id)
        hybrid_score = alpha * svd_score + (1 - alpha) * sim_movie_svd_score
        hybrid_scores.append((m_id, hybrid_score))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    recommended_movies = [movies[movies["movieId"] == m_id]["title"].values[0] for m_id, _ in hybrid_scores]
    return recommended_movies

st.title('🎬 Movie Recommender System')
st.markdown("""
    Enter a movie title and your user ID to get personalized recommendations!
""")

movie_input = st.text_input('🎥 Enter Movie Title:')
user_input = st.number_input('👤 Enter User ID:', min_value=1, max_value=1000, step=1)

if st.button('✨ Get Recommendations'):
    with st.spinner('⏳ Generating recommendations...'):
        if not movie_input.strip():
            st.error('❌ Please enter a valid movie title.')
        elif user_input <= 0:
            st.error('❌ Please enter a valid user ID.')
        else:
            movie_id = get_movie_id(movie_input)
            if movie_id is None:
                st.error(f'❌ Movie "{movie_input}" not found! Please try another title.')
            else:
                st.success('✅ Recommendations generated successfully!')

                # Hybrid Recommendations
                st.subheader('🌟 Hybrid Recommendations:')
                hybrid_recommendations = hybrid_recommend(user_input, movie_id, k=5)
                st.table(hybrid_recommendations)
