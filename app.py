import streamlit as st
import pandas as pd
import requests
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    web_series_df = pd.read_csv("All_Streaming_Shows.csv")
    return movies_df, web_series_df

def get_poster(title):
    api_key = "8db21eb4"  # OMDb API Key
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    response = requests.get(url).json()
    if response.get('Poster') and response['Poster'] != "N/A":
        return response['Poster']
    return "https://via.placeholder.com/500x750?text=No+Image"

def recommend_movies(title, movies_df, similarity):
    if title not in movies_df['title'].values:
        return "Movie not found in dataset."
    
    index = movies_df[movies_df['title'] == title].index[0]
    distances = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)[1:4]
    recommendations = [movies_df.iloc[i[0]] for i in distances]
    return recommendations

def recommend_web_series(title, web_series_df):
    if title not in web_series_df['Series Title'].values:
        return "Web Series not found in dataset."
    
    genre = web_series_df.loc[web_series_df['Series Title'] == title, 'Genre'].values[0]
    recommendations = web_series_df[web_series_df['Genre'] == genre].sample(n=3, replace=True)
    return recommendations

st.set_page_config(page_title="Movie & Web Series Recommendation", layout="wide")
st.markdown("""
    <style>
        .main-container {
            background-image: url('https://source.unsplash.com/1600x900/?cinema,movie');
            background-size: cover;
            background-position: center;
            color: white;
            text-align: center;
            font-family: Arial, sans-serif;
            padding: 50px 0;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #FFD700;
            margin-bottom: 20px;
        }
        .recommend-card {
            background-color: rgba(46, 46, 46, 0.9);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Movies", "Web Series", "About"])

if page == "Home":
    st.markdown("<div class='main-container'><div class='title'>Welcome to the Movie & Web Series Recommendation System</div></div>", unsafe_allow_html=True)
    st.write("Discover the best movies and web series based on your taste!")

elif page == "Movies":
    movies_df, web_series_df = load_data()
    with open("similarity.pkl", "rb") as file:
        movie_similarity = pickle.load(file)
    selected_movie = st.selectbox("Select a Movie:", movies_df['title'].unique())
    if st.button("Get Recommendations"):
        recommendations = recommend_movies(selected_movie, movies_df, movie_similarity)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            for movie in recommendations:
                st.markdown(f"<div class='recommend-card'><h3>{movie['title']}</h3></div>", unsafe_allow_html=True)
                st.image(get_poster(movie['title']), width=200)
                st.write(f"**Overview:** {movie['overview']}")

elif page == "Web Series":
    movies_df, web_series_df = load_data()
    selected_series = st.selectbox("Select a Web Series:", web_series_df['Series Title'].unique())
    if st.button("Get Recommendations"):
        recommendations = recommend_web_series(selected_series, web_series_df)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            for _, row in recommendations.iterrows():
                st.markdown(f"<div class='recommend-card'><h3>{row['Series Title']}</h3></div>", unsafe_allow_html=True)
                st.image(get_poster(row['Series Title']), width=200)
                st.write(f"**Genre:** {row['Genre']}")
                st.write(f"**Streaming on:** {row['Streaming Platform']}")

elif page == "About":
    st.title("About This Website")
    st.write("This is a recommendation system that helps users discover movies and web series based on their interests.")
    st.write("Built using Streamlit and powered by machine learning for intelligent recommendations.")
