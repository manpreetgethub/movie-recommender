# Movie Recommendation System with Beautiful UI
# Complete working code for VS Code

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
import random

# Page configuration
st.set_page_config(
    page_title="CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF4B4B, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
        border-left: 5px solid #FF4B4B;
        padding-left: 15px;
    }
    
    /* Cards */
    .movie-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #FF4B4B, #FF6B6B);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
        background: linear-gradient(45deg, #FF6B6B, #FF4B4B);
    }
    
    /* Input fields */
    .stSelectbox, .stMultiselect, .stSlider {
        background-color: white;
        border-radius: 15px;
        padding: 15px;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Load movie data
@st.cache_data
def load_movie_data():
    try:
        movies = pd.read_csv('movies.csv')
    except:
        # Create sample data if file doesn't exist
        movies = pd.DataFrame({
            'movie_id': range(1, 21),
            'title': [f'Movie {i}' for i in range(1, 21)],
            'genre': [random.choice(['Action', 'Drama', 'Comedy', 'Sci-Fi', 'Thriller', 'Crime']) for _ in range(20)],
            'rating': [round(random.uniform(6.0, 9.5), 1) for _ in range(20)],
            'year': [random.randint(1990, 2023) for _ in range(20)],
            'director': [f'Director {i}' for i in range(1, 21)],
            'description': [f'This is an amazing movie about {random.choice(["adventure", "love", "mystery", "action", "drama"])}. A must watch for all movie lovers!' for _ in range(20)]
        })
    return movies

# Recommendation engine
class MovieRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.feature_matrix = self._create_feature_matrix()
    
    def _create_feature_matrix(self):
        # Combine features for content-based filtering
        features = self.movies['genre'] + ' ' + self.movies['director'] + ' ' + self.movies['description'].fillna('')
        return self.vectorizer.fit_transform(features)
    
    def get_recommendations(self, liked_movies, n_recommendations=5):
        if not liked_movies:
            return self.movies.sample(min(n_recommendations, len(self.movies)))
        
        # Get indices of liked movies
        liked_indices = self.movies[self.movies['title'].isin(liked_movies)].index
        
        if len(liked_indices) == 0:
            return self.movies.sample(min(n_recommendations, len(self.movies)))
        
        # Calculate similarity
        liked_features = self.feature_matrix[liked_indices]
        avg_liked_features = liked_features.mean(axis=0)
        similarities = cosine_similarity(avg_liked_features, self.feature_matrix).flatten()
        
        # Get top recommendations
        similar_indices = similarities.argsort()[::-1]
        recommendations = []
        for idx in similar_indices:
            if idx not in liked_indices and self.movies.iloc[idx]['title'] not in liked_movies:
                recommendations.append(self.movies.iloc[idx])
            if len(recommendations) >= n_recommendations:
                break
        
        return pd.DataFrame(recommendations)

# Main application
def main():
    # Load data
    movies_df = load_movie_data()
    recommender = MovieRecommender(movies_df)
    
    # Sidebar with preferences
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1 style='color: white; font-size: 2rem;'>🎬 CineMatch</h1>
            <p style='color: #bdc3c7;'>Your Personal Movie Guide</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # User preferences
        st.subheader("🎯 Your Preferences")
        
        genres = sorted(movies_df['genre'].unique())
        selected_genres = st.multiselect(
            "Favorite Genres:",
            options=genres,
            default=['Action', 'Drama'],
            help="Select your favorite movie genres"
        )
        
        year_range = st.slider(
            "Release Year Range:",
            min_value=int(movies_df['year'].min()),
            max_value=int(movies_df['year'].max()),
            value=(2000, 2023),
            help="Select your preferred release years"
        )
        
        min_rating = st.slider(
            "Minimum Rating:",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.1,
            help="Set minimum rating filter"
        )
        
        n_recommendations = st.slider(
            "Number of Recommendations:",
            min_value=3,
            max_value=10,
            value=5,
            help="How many recommendations would you like?"
        )
        
        st.markdown("---")
        
        # User stats
        st.subheader("📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Movies", len(movies_df))
        with col2:
            st.metric("Avg Rating", f"{movies_df['rating'].mean():.1f}")
    
    # Main content
    st.markdown('<h1 class="main-header">🎥 CineMatch - Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    # Filter movies based on preferences
    filtered_movies = movies_df[
        (movies_df['genre'].isin(selected_genres)) &
        (movies_df['year'].between(year_range[0], year_range[1])) &
        (movies_df['rating'] >= min_rating)
    ].copy()
    
    # Display filtered movies
    st.markdown('<h2 class="section-header">🎯 Select Movies You Like</h2>', unsafe_allow_html=True)
    
    if len(filtered_movies) == 0:
        st.warning("No movies match your filters. Try adjusting your preferences.")
        filtered_movies = movies_df
    
    # Initialize session state for liked movies
    if 'liked_movies' not in st.session_state:
        st.session_state.liked_movies = []
    
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    
    # Display movies in a grid
    cols = st.columns(3)
    for idx, movie in filtered_movies.iterrows():
        col_idx = idx % 3
        with cols[col_idx]:
            with st.container():
                # Movie card
                st.markdown(f"""
                <div class="movie-card">
                    <h3 style='color: #2c3e50; margin-bottom: 10px;'>{movie['title']}</h3>
                    <p style='color: #7f8c8d; margin: 5px 0;'><strong>🎭 Genre:</strong> {movie['genre']}</p>
                    <p style='color: #7f8c8d; margin: 5px 0;'><strong>⭐ Rating:</strong> {movie['rating']}/10</p>
                    <p style='color: #7f8c8d; margin: 5px 0;'><strong>📅 Year:</strong> {movie['year']}</p>
                    <p style='color: #7f8c8d; margin: 5px 0;'><strong>🎬 Director:</strong> {movie['director']}</p>
                    <p style='color: #7f8c8d; margin-top: 10px;'>{movie['description'][:100]}...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Like button
                if st.button(f"❤️ Like", key=f"like_{movie['movie_id']}", use_container_width=True):
                    if movie['title'] not in st.session_state.liked_movies:
                        st.session_state.liked_movies.append(movie['title'])
                        st.success(f"Added '{movie['title']}' to your favorites!")
                        st.rerun()
    
    # Display liked movies
    if st.session_state.liked_movies:
        st.markdown("---")
        st.markdown('<h2 class="section-header">❤️ Your Favorite Movies</h2>', unsafe_allow_html=True)
        
        liked_cols = st.columns(3)
        for i, movie_title in enumerate(st.session_state.liked_movies):
            with liked_cols[i % 3]:
                movie_info = movies_df[movies_df['title'] == movie_title].iloc[0]
                st.markdown(f"""
                <div class="movie-card" style='background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);'>
                    <h4 style='color: #2d3436;'>{movie_info['title']}</h4>
                    <p style='color: #636e72;'>{movie_info['genre']} • ⭐ {movie_info['rating']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Get recommendations button
        if st.button("🎯 Get Personalized Recommendations", use_container_width=True):
            st.session_state.show_recommendations = True
    
    # Recommendation section
    if st.session_state.show_recommendations and st.session_state.liked_movies:
        st.markdown("---")
        st.markdown('<h2 class="section-header">🎉 Personalized Recommendations</h2>', unsafe_allow_html=True)
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            st.session_state.liked_movies, 
            n_recommendations
        )
        
        # Display recommendations
        for idx, movie in recommendations.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3 style='color: white; margin-bottom: 15px;'>#{idx + 1}: {movie['title']}</h3>
                    <p style='color: #ecf0f1; margin: 8px 0;'><strong>🎭 Genre:</strong> {movie['genre']}</p>
                    <p style='color: #ecf0f1; margin: 8px 0;'><strong>⭐ Rating:</strong> {movie['rating']}/10</p>
                    <p style='color: #ecf0f1; margin: 8px 0;'><strong>📅 Year:</strong> {movie['year']}</p>
                    <p style='color: #ecf0f1; margin: 8px 0;'><strong>🎬 Director:</strong> {movie['director']}</p>
                    <p style='color: #ecf0f1; margin-top: 15px;'>{movie['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Statistics section
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Movie Database Insights</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Genre distribution
        genre_counts = movies_df['genre'].value_counts()
        fig1 = px.pie(
            values=genre_counts.values, 
            names=genre_counts.index, 
            title="🎭 Genre Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Rating distribution
        fig2 = px.histogram(
            movies_df, 
            x='rating', 
            nbins=15, 
            title="⭐ Rating Distribution",
            color_discrete_sequence=['#FF4B4B']
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        # Yearly movie count
        year_counts = movies_df['year'].value_counts().sort_index()
        fig3 = px.bar(
            x=year_counts.index, 
            y=year_counts.values, 
            title="📅 Movies by Release Year",
            labels={'x': 'Year', 'y': 'Number of Movies'},
            color=year_counts.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>🎬 Built with ❤️ using Streamlit • CineMatch Movie Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()