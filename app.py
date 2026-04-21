import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import ast
import re
import requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="MovieFinder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0d1117; }

[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #e6edf3;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #30363d;
    margin-bottom: 1rem;
}

[data-testid="stTextInput"] input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #f0a500 !important;
    box-shadow: 0 0 0 3px rgba(240, 165, 0, 0.15) !important;
}
[data-testid="stTextInput"] label { color: #8b949e !important; font-size: 0.85rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #f0a500, #e08000) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

.poster-card {
    border-radius: 10px;
    overflow: hidden;
    position: relative;
}
.poster-card img {
    width: 100%;
    border-radius: 10px;
    display: block;
    aspect-ratio: 2/3;
    object-fit: cover;
    background: #161b22;
    transition: transform 0.25s ease, filter 0.25s ease;
}
.poster-card img:hover {
    transform: scale(1.04);
    filter: brightness(1.1);
}
.poster-title {
    color: #e6edf3;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.4rem 0 0.1rem 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.poster-meta {
    color: #8b949e;
    font-size: 0.75rem;
    margin-bottom: 0.4rem;
}
.feat-genre-tag {
    display: inline-block;
    background-color: rgba(240, 165, 0, 0.1);
    color: #c8860a;
    border: 1px solid rgba(240, 165, 0, 0.18);
    padding: 0.1rem 0.4rem;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 500;
    margin: 0 0.15rem 0.25rem 0;
}

.section-label {
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
}
.section-title {
    color: #e6edf3;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
}
.section-sub {
    color: #8b949e;
    font-size: 0.88rem;
    margin-bottom: 1.5rem;
}

.query-banner {
    background: linear-gradient(135deg, rgba(240, 165, 0, 0.12), rgba(240, 165, 0, 0.04));
    border: 1px solid rgba(240, 165, 0, 0.25);
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 2rem;
}
.query-banner .q-label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
.query-banner .q-title { color: #f0a500; font-size: 1.7rem; font-weight: 700; margin: 0.25rem 0 0.5rem 0; }
.query-banner .q-overview {
    color: #c9d1d9; font-size: 0.88rem; line-height: 1.6;
    margin-top: 0.5rem; font-style: italic;
}
.genre-tag {
    display: inline-block;
    background-color: rgba(240, 165, 0, 0.12);
    color: #f0a500;
    border: 1px solid rgba(240, 165, 0, 0.2);
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-size: 0.72rem; font-weight: 500; margin: 0 0.2rem 0.3rem 0;
}

.dim-divider { border: none; border-top: 1px solid #21262d; margin: 2rem 0; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER_IMG = "https://placehold.co/342x513/161b22/444c56?text=No+Image"

# Inline SVG placeholder — always works, no external request needed
# Must use literal # and % here (not URL-encoded) because content is base64-encoded, not URL-encoded
_svg = (
    b"<svg xmlns='http://www.w3.org/2000/svg' width='342' height='513'>"
    b"<rect width='342' height='513' fill='#21262d' rx='10'/>"
    b"<text x='171' y='230' font-family='sans-serif' font-size='48' fill='#30363d' text-anchor='middle'>&#127909;</text>"
    b"<text x='171' y='278' font-family='sans-serif' font-size='13' fill='#8b949e' text-anchor='middle'>No Poster</text>"
    b"</svg>"
)
PLACEHOLDER_B64 = "data:image/svg+xml;base64," + base64.b64encode(_svg).decode()


@st.cache_resource
def load_models():
    tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    indices = pickle.load(open('indices.pkl', 'rb'))
    df = pd.read_pickle('df.pkl')
    try:
        meta = pd.read_csv(
            'movies_metadata.csv',
            usecols=['id', 'poster_path', 'vote_count'],
            low_memory=False
        )
        meta['id'] = meta['id'].astype(str).str.strip()
        df['id'] = df['id'].astype(str).str.strip()
        df = df.merge(meta.drop_duplicates('id'), on='id', how='left')
    except Exception:
        df['poster_path'] = None
        df['vote_count'] = 0
    return tfidf_matrix, indices, df


@st.cache_data(ttl=3600)
def fetch_poster_b64(url: str) -> str:
    """Fetch a TMDb poster URL and return as base64 data URI."""
    try:
        r = requests.get(url, timeout=4, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            ct = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            return f"data:{ct};base64,{base64.b64encode(r.content).decode()}"
    except Exception:
        pass
    return ""


@st.cache_data(ttl=86400)
def fetch_wiki_image_b64(title: str, year=None) -> str:
    """Query Wikipedia's free REST API to get a movie image when no TMDb poster exists."""
    _headers = {"User-Agent": "MovieFinder/1.0 (educational project)"}
    candidates = []
    if year:
        candidates.append(f"{title} ({year} film)")
    candidates += [f"{title} film", title]
    for q in candidates:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(q)}"
            r = requests.get(url, timeout=4, headers=_headers)
            if r.status_code == 200:
                thumb = r.json().get("thumbnail", {}).get("source", "")
                if thumb:
                    img_r = requests.get(thumb, timeout=4, headers=_headers)
                    if img_r.status_code == 200:
                        ct = img_r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                        return f"data:{ct};base64,{base64.b64encode(img_r.content).decode()}"
        except Exception:
            continue
    return PLACEHOLDER_B64


def get_img_src(movie: dict) -> str:
    """Return a base64 data URI for a movie dict with 'poster_url', 'title', 'year'."""
    poster_url = movie.get("poster_url", "")
    title = movie.get("title", "")
    year = movie.get("year")
    # 1. Try TMDb poster path
    if poster_url and poster_url != PLACEHOLDER_IMG:
        result = fetch_poster_b64(poster_url)
        if result:
            return result
    # 2. Fall back to Wikipedia
    if title:
        return fetch_wiki_image_b64(title, year)
    return PLACEHOLDER_B64


def prefetch_posters(movies: list) -> None:
    """Fetch all posters in parallel (with Wikipedia fallback) to warm st.cache_data."""
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(get_img_src, m) for m in movies]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass


def _notna(val):
    try:
        return bool(pd.notna(val))
    except (TypeError, ValueError):
        return True


def parse_genres(genre_val):
    try:
        if pd.isna(genre_val):
            return []
    except (TypeError, ValueError):
        pass
    if genre_val is None or genre_val == '':
        return []
    if isinstance(genre_val, list):
        if genre_val and isinstance(genre_val[0], dict):
            return [g['name'] for g in genre_val if isinstance(g, dict) and 'name' in g]
        return [str(g) for g in genre_val if g]
    try:
        items = ast.literal_eval(str(genre_val))
        if isinstance(items, list):
            if items and isinstance(items[0], dict):
                return [g['name'] for g in items if isinstance(g, dict) and 'name' in g]
            return [str(g) for g in items if g]
    except Exception:
        pass
    names = re.findall(r"'name':\s*'([^']+)'", str(genre_val))
    return names if names else []


def extract_year(date_val):
    try:
        if pd.isna(date_val):
            return None
    except (TypeError, ValueError):
        pass
    m = re.match(r'(\d{4})', str(date_val).strip())
    return int(m.group(1)) if m else None


def find_closest_title(user_input, all_titles, cutoff=0.6):
    string_titles = [str(t) for t in all_titles if pd.notna(t)]
    matches = difflib.get_close_matches(user_input, string_titles, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def get_poster_url(movie_row):
    has_col = 'poster_path' in movie_row.index
    if has_col:
        p = movie_row['poster_path']
        if _notna(p) and str(p).strip() not in ('', 'nan', 'None'):
            return f"{TMDB_IMG_BASE}{str(p).strip()}"
    return PLACEHOLDER_IMG


@st.cache_data
def get_featured_movies(n=8):
    rows = df.iloc[pd.to_numeric(df['popularity'], errors='coerce').fillna(0).nlargest(n * 4).index]
    result = []
    for _, movie in rows.iterrows():
        poster_url = get_poster_url(movie)
        genres = parse_genres(movie.get('genres', ''))
        year = extract_year(movie.get('release_date', ''))
        rating = float(movie['vote_average']) if _notna(movie.get('vote_average')) else 0.0
        result.append({
            'title': str(movie['title']),
            'poster_url': poster_url,
            'genres': genres[:2],
            'year': year,
            'rating': rating,
        })
        if len(result) >= n:
            break
    return result


def get_recommendations(title, n=30):
    if title not in indices.index:
        closest = find_closest_title(title, indices.index.tolist())
        if closest is None:
            return None, None, f"No match found for '{title}'. Try a different title."
        title_used = closest
    else:
        title_used = title

    idx = int(indices.loc[title_used, 0])
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n + 1]

    recs = []
    for i in similar_idx:
        movie = df.iloc[i]
        genres = parse_genres(movie.get('genres', ''))
        year = extract_year(movie.get('release_date', ''))
        recs.append({
            'title': str(movie['title']),
            'rating': float(movie['vote_average']) if _notna(movie.get('vote_average')) else 0.0,
            'year': year,
            'genres': genres,
            'similarity': float(sim_scores[i]),
            'poster_url': get_poster_url(movie),
        })

    query_row = df[df['title'] == title_used]
    query_info = None
    if not query_row.empty:
        qm = query_row.iloc[0]
        query_info = {
            'title': title_used,
            'year': extract_year(qm.get('release_date', '')),
            'rating': float(qm['vote_average']) if _notna(qm.get('vote_average')) else 0.0,
            'genres': parse_genres(qm.get('genres', '')),
            'overview': str(qm.get('overview', '')) if _notna(qm.get('overview')) else '',
            'poster_url': get_poster_url(qm),
        }

    return recs, query_info, title_used


# ── Load ───────────────────────────────────────────────────────────────────────
try:
    tfidf_matrix, indices, df = load_models()
    models_loaded = True
    all_years = df['release_date'].apply(extract_year).dropna().astype(int)
    min_year_data = int(all_years.min()) if len(all_years) else 1900
    max_year_data = int(all_years.max()) if len(all_years) else 2024
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False
    min_year_data, max_year_data = 1900, 2024

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ('recommendations', []),
    ('query_info', None),
    ('error_message', ""),
    ('last_search', ""),
    ('featured_movie', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Consume featured card click before rendering input ─────────────────────────
featured_trigger = st.session_state.featured_movie
if featured_trigger:
    st.session_state.featured_movie = None
    st.session_state['_search_val'] = featured_trigger

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    num_results = st.select_slider("Results to show", options=[8, 12, 16, 20, 24], value=12)
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, 0.5, format="%.1f ⭐")
    year_range = st.slider("Release year", min_year_data, max_year_data, (min_year_data, max_year_data))
    sort_by = st.selectbox("Sort by", ["Similarity", "Rating (high → low)", "Year (new → old)", "Year (old → new)"])
    st.markdown("---")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.78rem;'>TF-IDF content similarity on plot, genres & taglines</p>",
        unsafe_allow_html=True
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:2.5rem 0 2rem 0;text-align:center;'>
    <div style='font-size:2.8rem;font-weight:800;color:#e6edf3;letter-spacing:-0.02em;'>🎬 MovieFinder</div>
    <div style='color:#8b949e;font-size:1rem;margin-top:0.4rem;'>Discover movies you'll love — powered by content similarity</div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("Failed to load models. Make sure all pickle files are in the same directory.")
    st.stop()

# ── Search bar ─────────────────────────────────────────────────────────────────
input_val = st.session_state.pop('_search_val', None)
if input_val:
    st.session_state['search_widget'] = input_val

col_input, col_btn = st.columns([5, 1])
with col_input:
    search_movie = st.text_input(
        "Movie title",
        placeholder="e.g. Inception, The Dark Knight, Toy Story…",
        label_visibility="collapsed",
        key="search_widget",
    )
with col_btn:
    search_button = st.button("Search", use_container_width=True)

# ── Trigger search ─────────────────────────────────────────────────────────────
trigger = bool(featured_trigger) or search_button or (
    search_movie.strip() and search_movie.strip() != st.session_state.last_search
)
query_term = (featured_trigger or search_movie).strip()

if trigger and query_term:
    with st.spinner("Finding similar movies…"):
        recs, query_info, result = get_recommendations(query_term)
    if recs is not None:
        st.session_state.recommendations = recs
        st.session_state.query_info = query_info
        st.session_state.error_message = ""
        st.session_state.last_search = query_term
    else:
        st.session_state.recommendations = []
        st.session_state.query_info = None
        st.session_state.error_message = result
        st.session_state.last_search = query_term

# ── Error state ────────────────────────────────────────────────────────────────
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    st.markdown("**Try one of these popular titles:**")
    top_movies = df.iloc[pd.to_numeric(df['popularity'], errors='coerce').fillna(0).nlargest(6).index]['title'].values.flatten()
    for i, title in enumerate(st.columns(6)):
        with title:
            st.caption(f"🎬 {top_movies[i]}")

# ── Results ────────────────────────────────────────────────────────────────────
elif st.session_state.recommendations:
    q = st.session_state.query_info

    # Query banner
    if q:
        genre_tags = "".join(f"<span class='genre-tag'>{g}</span>" for g in q['genres'][:5])
        year_str = f" · {q['year']}" if q['year'] else ""
        rating_str = f" · ⭐ {q['rating']:.1f}" if q['rating'] > 0 else ""
        overview_html = (
            f"<div class='q-overview'>\"{q['overview'][:240]}{'…' if len(q['overview']) > 240 else ''}\"</div>"
            if q['overview'] else ""
        )
        bc1, bc2 = st.columns([1, 4])
        with bc1:
            img_src = get_img_src(q)
            st.markdown(
                f"<img src='{img_src}' style='width:100%;border-radius:10px;display:block;aspect-ratio:2/3;object-fit:cover;background:#161b22;'>",
                unsafe_allow_html=True
            )
        with bc2:
            st.markdown(f"""
            <div class='query-banner' style='height:100%;'>
                <div class='q-label'>Because you searched for</div>
                <div class='q-title'>{q['title']}</div>
                <div>{genre_tags}<span style='color:#8b949e;font-size:0.82rem;'>{year_str}{rating_str}</span></div>
                {overview_html}
            </div>
            """, unsafe_allow_html=True)

    # Filters
    filtered = [
        r for r in st.session_state.recommendations
        if r['rating'] >= min_rating
        and (r['year'] is None or year_range[0] <= r['year'] <= year_range[1])
    ]
    if sort_by == "Rating (high → low)":
        filtered.sort(key=lambda x: x['rating'], reverse=True)
    elif sort_by == "Year (new → old)":
        filtered.sort(key=lambda x: x['year'] or 0, reverse=True)
    elif sort_by == "Year (old → new)":
        filtered.sort(key=lambda x: x['year'] or 9999)

    displayed = filtered[:num_results]
    filter_note = f" (filtered from {len(st.session_state.recommendations)})" if (
        min_rating > 0 or year_range != (min_year_data, max_year_data)
    ) else ""

    st.markdown(f"<div class='section-label'>{len(displayed)} recommendations{filter_note}</div>", unsafe_allow_html=True)

    if not displayed:
        st.warning("No results match your filters. Try relaxing the rating or year range.")
    else:
        with st.spinner("Loading posters…"):
            prefetch_posters(displayed)
        cols_per_row = 4
        cols = st.columns(cols_per_row, gap="medium")
        for rank, movie in enumerate(displayed):
            with cols[rank % cols_per_row]:
                img_src = get_img_src(movie)
                year_str = str(movie['year']) if movie['year'] else ''
                meta = f"{year_str} · ★ {movie['rating']:.1f}" if year_str else (f"★ {movie['rating']:.1f}" if movie['rating'] > 0 else '')
                st.markdown(f"""
                <div class='poster-card'>
                    <img src='{img_src}'>
                </div>
                <div class='poster-title' title='{movie["title"]}'>{movie["title"]}</div>
                <div class='poster-meta'>{meta}</div>
                """, unsafe_allow_html=True)
                if st.button("Find Similar →", key=f"rec_{rank}_{movie['title']}", use_container_width=True):
                    st.session_state.featured_movie = movie['title']
                    st.rerun()

# ── Landing / Featured movies ──────────────────────────────────────────────────
else:
    st.markdown("<hr class='dim-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-title'>Popular Right Now</div>
    <div class='section-sub'>Click any movie to instantly get recommendations</div>
    """, unsafe_allow_html=True)

    featured = get_featured_movies(8)
    with st.spinner("Loading posters…"):
        prefetch_posters(featured)
    row1_cols = st.columns(4, gap="medium")
    row2_cols = st.columns(4, gap="medium")
    all_cols = row1_cols + row2_cols

    for i, (col, movie) in enumerate(zip(all_cols, featured)):
        with col:
            img_src = get_img_src(movie)
            genre_tags = "".join(
                f"<span class='feat-genre-tag'>{g}</span>" for g in movie['genres']
            )
            year_str = str(movie['year']) if movie['year'] else ""
            meta = f"{year_str} · ★ {movie['rating']:.1f}" if year_str else f"★ {movie['rating']:.1f}"
            st.markdown(f"""
            <div class='poster-card'>
                <img src='{img_src}'>
            </div>
            <div style='padding:0.4rem 0.1rem 0.1rem 0.1rem;'>
                <div class='poster-title' title='{movie['title']}'>{movie['title']}</div>
                <div class='poster-meta'>{meta}</div>
                <div>{genre_tags}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Find Similar →", key=f"feat_{i}", use_container_width=True):
                st.session_state.featured_movie = movie['title']
                st.rerun()
