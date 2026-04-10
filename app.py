import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Configure page
st.set_page_config(
    page_title="🎬 MovieFinder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    :root {
        --primary-color: #FFB81C;
        --dark-bg: #0F1419;
        --card-bg: #1a1f2e;
        --text-primary: #ffffff;
        --text-secondary: #b0b8c1;
    }
    
    .main {
        background-color: var(--dark-bg);
        color: var(--text-primary);
    }
    
    .stMetric {
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 184, 28, 0.1);
    }
    
    .movie-card {
        background: linear-gradient(135deg, rgba(255, 184, 28, 0.05) 0%, rgba(255, 184, 28, 0.01) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 184, 28, 0.15);
        transition: all 0.3s ease;
    }
    
    .movie-card:hover {
        border-color: rgba(255, 184, 28, 0.4);
        background: linear-gradient(135deg, rgba(255, 184, 28, 0.12) 0%, rgba(255, 184, 28, 0.05) 100%);
        box-shadow: 0 8px 32px rgba(255, 184, 28, 0.1);
    }
    
    .rating-badge {
        display: inline-block;
        background-color: rgba(255, 184, 28, 0.2);
        color: #FFB81C;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .popularity-badge {
        display: inline-block;
        background-color: rgba(100, 200, 255, 0.2);
        color: #64c8ff;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-left: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load pickled models and data"""
    tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    indices = pickle.load(open('indices.pkl', 'rb'))
    df = pd.read_pickle('df.pkl')
    return tfidf_matrix, indices, df

try:
    tfidf_matrix, indices, df = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.error("Make sure all pickle files are in the same directory as this script.")
    models_loaded = False

def find_closest_title(user_input, all_titles, cutoff=0.6):
    """Find the closest matching title using difflib"""
    string_titles = [str(t) for t in all_titles if pd.notna(t)]
    if not string_titles:
        return None
    matches = difflib.get_close_matches(user_input, string_titles, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def get_recommendations(title, n=12):
    """Get movie recommendations based on title"""
    # Use indices from pickle which maps to tfidf_matrix correctly
    if title not in indices.index:
        closest = find_closest_title(title, indices.index.tolist())
        if closest is None:
            return None, f"Movie '{title}' not found. Try another title!"
        title_used = closest
    else:
        title_used = title

    # Get the index from the indices dataframe
    idx = int(indices.loc[title_used, 0])
    
    # Get recommendations
    sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_score.argsort()[::-1][1:n+1]
    
    recommendations = []
    for i in similar_idx:
        movie = df.iloc[i]
        recommendations.append({
            'title': str(movie['title']),
            'rating': float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0,
            'popularity': float(movie['popularity']) if pd.notna(movie['popularity']) else 0,
            'id': str(movie['id']),
            'release_date': str(movie['release_date']) if pd.notna(movie['release_date']) else 'N/A'
        })
    
    return recommendations, title_used

# Header
st.markdown("""
<div style='text-align: center; margin-bottom: 3rem;'>
    <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>🎬 MovieFinder</h1>
    <p style='font-size: 1.2rem; color: var(--text-secondary); margin: 0;'>Discover movies you'll love</p>
    <p style='font-size: 0.95rem; color: var(--text-secondary); margin: 0.5rem 0 0 0;'>Powered by content-based recommendations</p>
</div>
""", unsafe_allow_html=True)

if models_loaded:
    # Search section
    col1, col2 = st.columns([4, 1])
    with col1:
        search_movie = st.text_input(
            "Search for a movie",
            placeholder="Try: Avengers, Inception, Toy Story...",
            key="search_input"
        )
    with col2:
        search_button = st.button("🔍 Find", use_container_width=True)

    # Initialize session state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
        st.session_state.query_movie = ""
        st.session_state.error_message = ""

    # Handle search
    if search_button or (search_movie and search_movie != st.session_state.get('last_search', '')):
        if search_movie.strip():
            st.session_state.last_search = search_movie
            recommendations, result = get_recommendations(search_movie.strip())
            
            if recommendations:
                st.session_state.recommendations = recommendations
                st.session_state.query_movie = result
                st.session_state.error_message = ""
            else:
                st.session_state.recommendations = []
                st.session_state.error_message = result
                st.session_state.query_movie = ""

    # Display results
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        
        # Show some popular movies as suggestions
        st.info("💡 **Suggestions:** Try searching for popular movies like:")
        popular_samples = df.nlargest(5, 'vote_average')[['title', 'vote_average']].to_dict('records')
        cols = st.columns(5)
        for i, movie in enumerate(popular_samples):
            with cols[i]:
                st.caption(f"⭐ {movie['title']}")

    elif st.session_state.recommendations:
        # Display query movie info
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(255, 184, 28, 0.1) 0%, rgba(255, 184, 28, 0.02) 100%); 
                    padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255, 184, 28, 0.2); margin-bottom: 2rem;'>
            <p style='margin: 0; color: var(--text-secondary); font-size: 0.95rem;'>Based on:</p>
            <h2 style='margin: 0.5rem 0 0 0; color: #FFB81C; font-size: 1.8rem;'>{st.session_state.query_movie}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Display recommendations in grid
        st.markdown(f"#### Found {len(st.session_state.recommendations)} similar movies")
        
        cols = st.columns(3)
        for idx, movie in enumerate(st.session_state.recommendations):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class='movie-card'>
                    <div style='margin-bottom: 0.8rem;'>
                        <h4 style='margin: 0 0 0.5rem 0; color: var(--text-primary); font-size: 1.1rem;'>{movie['title']}</h4>
                        <p style='margin: 0; color: var(--text-secondary); font-size: 0.85rem;'>{movie['release_date']}</p>
                    </div>
                    <div style='display: flex; gap: 0.5rem; align-items: center;'>
                        <span class='rating-badge'>⭐ {movie['rating']:.1f}</span>
                        <span class='popularity-badge'>📈 {movie['popularity']:.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        # Stats summary
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_rating = sum(m['rating'] for m in st.session_state.recommendations) / len(st.session_state.recommendations)
            st.metric("Average Rating", f"{avg_rating:.2f}/10", "⭐")
        with col2:
            avg_popularity = sum(m['popularity'] for m in st.session_state.recommendations) / len(st.session_state.recommendations)
            st.metric("Average Popularity", f"{avg_popularity:.1f}", "📈")
        with col3:
            st.metric("Total Results", len(st.session_state.recommendations), "🎬")

else:
    st.error("Failed to load the recommendation model. Please check that all required pickle files are present.")