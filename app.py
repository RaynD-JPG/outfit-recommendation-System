"""
StyleMatch — Personalized Outfit Recommendation System
Streamlit Web App

Run with:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from user_profile import UserProfileBuilder
from recommendation import OutfitRecommender

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StyleMatch",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .stApp { background: #f7f8fc; }
  .block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1200px; }

  /* ── Navbar ── */
  .navbar {
    background: #111827;
    padding: 16px 28px;
    border-radius: 0 0 16px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
  }
  .navbar-logo { font-size: 22px; font-weight: 800; color: #fff; letter-spacing: 0.5px; }
  .navbar-logo span { color: #f43f5e; }
  .navbar-tagline { color: #6b7280; font-size: 13px; }

  /* ── Hero ── */
  .hero {
    background: linear-gradient(135deg, #111827 0%, #1e3a5f 100%);
    border-radius: 20px;
    padding: 56px 48px;
    text-align: center;
    margin-bottom: 36px;
    color: #fff;
  }
  .hero-pill {
    display: inline-block;
    background: rgba(244,63,94,0.15);
    color: #f43f5e;
    border: 1px solid rgba(244,63,94,0.4);
    border-radius: 999px;
    padding: 5px 18px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 20px;
  }
  .hero h1 { font-size: 40px; font-weight: 800; line-height: 1.2; margin-bottom: 14px; }
  .hero h1 em { color: #f43f5e; font-style: normal; }
  .hero p { color: #94a3b8; font-size: 16px; max-width: 480px; margin: 0 auto; line-height: 1.6; }

  /* ── Cards ── */
  .card {
    background: #fff;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 24px;
  }
  .card-title {
    font-size: 15px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid #f1f5f9;
  }

  /* ── Steps ── */
  .steps-row { display: flex; gap: 16px; margin-bottom: 32px; }
  .step {
    flex: 1;
    background: #fff;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .step-circle {
    width: 38px; height: 38px;
    background: #f43f5e;
    color: #fff;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 16px;
    margin: 0 auto 12px;
  }
  .step h4 { font-size: 14px; font-weight: 700; color: #111827; margin-bottom: 6px; }
  .step p { font-size: 12px; color: #6b7280; line-height: 1.5; }

  /* ── Gender toggle ── */
  .gender-row { display: flex; gap: 12px; margin-bottom: 20px; }
  .gender-btn {
    flex: 1;
    padding: 14px;
    border-radius: 12px;
    border: 2px solid #e5e7eb;
    background: #fff;
    cursor: pointer;
    text-align: center;
    font-size: 15px;
    font-weight: 600;
    color: #374151;
    transition: all .15s;
  }
  .gender-btn.active {
    border-color: #f43f5e;
    background: #fff1f3;
    color: #f43f5e;
  }

  /* ── Results header ── */
  .results-header {
    background: linear-gradient(135deg, #111827, #1e3a5f);
    border-radius: 16px;
    padding: 28px 32px;
    color: #fff;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .results-header h2 { font-size: 22px; font-weight: 800; margin-bottom: 4px; }
  .results-header p { color: #94a3b8; font-size: 13px; }
  .results-badge {
    background: #f43f5e;
    color: #fff;
    border-radius: 999px;
    padding: 6px 18px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
  }

  /* ── Outfit cards ── */
  .outfit-card {
    background: #fff;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
    transition: transform .2s, box-shadow .2s;
    height: 100%;
  }
  .outfit-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.1);
  }
  .outfit-rank-bar {
    background: #111827;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 700;
    color: #f43f5e;
    letter-spacing: 0.5px;
  }
  .outfit-meta { padding: 10px 12px 14px; }
  .outfit-name {
    font-size: 12px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .score-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .score-bar-outer { flex: 1; height: 4px; background: #f1f5f9; border-radius: 2px; overflow: hidden; }
  .score-bar-inner { height: 100%; background: #f43f5e; border-radius: 2px; }
  .score-text { font-size: 11px; font-weight: 700; color: #f43f5e; min-width: 36px; text-align: right; }
  .outfit-tags { display: flex; gap: 5px; margin-top: 6px; }
  .tag {
    background: #f1f5f9;
    color: #6b7280;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
  }

  /* ── Top pick ── */
  .top-pick-card {
    background: #fff;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.05);
    border-top: 3px solid #f43f5e;
    margin-bottom: 28px;
  }
  .top-pick-badge {
    display: inline-block;
    background: #f43f5e;
    color: #fff;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }
  .top-pick-name { font-size: 22px; font-weight: 800; color: #111827; margin-bottom: 8px; }
  .top-pick-desc { font-size: 13px; color: #6b7280; margin-bottom: 18px; line-height: 1.5; }
  .stats-row { display: flex; gap: 28px; }
  .stat-item { text-align: center; }
  .stat-val { font-size: 24px; font-weight: 800; color: #111827; }
  .stat-lbl { font-size: 11px; color: #9ca3af; margin-top: 2px; }

  /* ── Tip box ── */
  .tip {
    background: #f8faff;
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #374151;
    margin-top: 16px;
    line-height: 1.5;
  }

  /* ── Streamlit overrides ── */
  div[data-testid="stFileUploader"] {
    border: 2px dashed #e5e7eb !important;
    border-radius: 14px !important;
    padding: 12px !important;
    background: #fafafa !important;
  }
  div[data-testid="stFileUploader"]:hover { border-color: #f43f5e !important; }

  .stButton > button {
    background: linear-gradient(135deg, #f43f5e, #be123c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  div[data-testid="stProgress"] > div > div {
    background: #f43f5e !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_recommender():
    r = OutfitRecommender()
    r.load_database()
    return r

@st.cache_resource(show_spinner=False)
def load_builder():
    return UserProfileBuilder()


# ── Session state ─────────────────────────────────────────────────────────────

defaults = {'page': 'upload', 'results': None, 'uploaded_images': [], 'gender': 'men'}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_uploaded_files(uploaded_files):
    paths = []
    for uf in uploaded_files:
        uf.seek(0)
        suffix = Path(uf.name).suffix or '.jpg'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uf.read())
        tmp.close()
        paths.append(tmp.name)
    return paths

def load_pil(path: str) -> Image.Image:
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

def page_upload():
    st.markdown("""
    <div class="navbar">
      <div class="navbar-logo">Style<span>Match</span></div>
      <div class="navbar-tagline">AI Outfit Recommendation • ResNet50 • Polyvore</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <div class="hero-pill">AI POWERED • RESNET50</div>
      <h1>Outfits Matched to<br/><em>Your Style</em></h1>
      <p>Upload clothing photos you love and we'll find the best matching outfits from over 16,000 curated looks.</p>
    </div>""", unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="steps-row">
      <div class="step">
        <div class="step-circle">1</div>
        <h4>Choose Your Gender</h4>
        <p>Select Men or Women so we only show you relevant outfits.</p>
      </div>
      <div class="step">
        <div class="step-circle">2</div>
        <h4>Upload Your Style</h4>
        <p>Upload 5–10 photos of clothes or outfits you already like.</p>
      </div>
      <div class="step">
        <div class="step-circle">3</div>
        <h4>Get Recommendations</h4>
        <p>Our AI finds your top 10 matching outfits instantly.</p>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Gender selector ──────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👤 Select Your Style Category</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        men_active = 'active' if st.session_state.gender == 'men' else ''
        if st.button("👔  Men's Fashion", key='btn_men'):
            st.session_state.gender = 'men'
            st.rerun()
    with col2:
        if st.button("👗  Women's Fashion", key='btn_women'):
            st.session_state.gender = 'women'
            st.rerun()

    gender_label = "Men's" if st.session_state.gender == 'men' else "Women's"
    st.success(f"✓ Showing **{gender_label}** outfits")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🖼️ Upload Your Style Photos</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag & drop photos here, or click to browse",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True,
        label_visibility='visible',
    )

    if uploaded_files:
        n = len(uploaded_files)
        if n < 3:
            st.warning(f"Uploaded {n} image{'s' if n > 1 else ''}. Upload at least 3 for best results.")
        else:
            st.success(f"✓ {n} image{'s' if n > 1 else ''} ready")

        # Preview — 5 per row
        display_files = uploaded_files[:10]
        rows = [display_files[i:i+5] for i in range(0, len(display_files), 5)]
        for row in rows:
            cols = st.columns(5)
            for i, uf in enumerate(row):
                with cols[i]:
                    uf.seek(0)
                    try:
                        img = Image.open(uf).convert('RGB')
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.caption("⚠ Can't preview")

    st.markdown("""
    <div class="tip">
      <strong>💡 Tip:</strong> Mix different item types — tops, trousers, shoes, jackets.
      The more variety, the better the AI understands your complete style.
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Submit ───────────────────────────────────────────────────────────────
    if uploaded_files and len(uploaded_files) >= 1:
        if st.button(f"✨  Find My {gender_label} Outfit Matches"):
            with st.spinner("Analysing your style... ⏳"):
                try:
                    img_paths = save_uploaded_files(uploaded_files)
                    builder = load_builder()
                    profile = builder.build_profile(img_paths)

                    if profile is None:
                        st.error("Feature extraction failed. Please try clearer photos with plain backgrounds.")
                        return

                    recommender = load_recommender()
                    results = recommender.recommend(
                        profile,
                        top_k=10,
                        gender=st.session_state.gender,
                        min_images=3,
                    )

                    if not results:
                        st.error("No matching outfits found. Try different photos.")
                        return

                    # Cache uploaded images as PIL
                    pil_imgs = []
                    for p in img_paths:
                        img = load_pil(p)
                        if img:
                            pil_imgs.append(img)

                    st.session_state.results = results
                    st.session_state.uploaded_images = pil_imgs
                    st.session_state.page = 'results'

                    for p in img_paths:
                        try: os.unlink(p)
                        except: pass

                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def page_results():
    results = st.session_state.results
    uploaded_images = st.session_state.uploaded_images
    gender_label = "Men's" if st.session_state.gender == 'men' else "Women's"

    st.markdown("""
    <div class="navbar">
      <div class="navbar-logo">Style<span>Match</span></div>
      <div class="navbar-tagline">AI Outfit Recommendation • ResNet50 • Polyvore</div>
    </div>""", unsafe_allow_html=True)

    if st.button("← Upload New Photos"):
        st.session_state.page = 'upload'
        st.session_state.results = None
        st.session_state.uploaded_images = []
        st.rerun()

    st.markdown(f"""
    <div class="results-header">
      <div>
        <h2>Your {gender_label} Recommendations</h2>
        <p>{len(uploaded_images)} style photos analysed &nbsp;•&nbsp; 16,722 outfits searched &nbsp;•&nbsp; Top 10 shown</p>
      </div>
      <div class="results-badge">{gender_label.upper()} • TOP 10</div>
    </div>""", unsafe_allow_html=True)

    # ── Uploaded style strip ─────────────────────────────────────────────────
    if uploaded_images:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📸 Your Style Reference</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(uploaded_images), 7))
        for i, img in enumerate(uploaded_images[:7]):
            with cols[i]:
                st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if not results:
        st.warning("No results to display.")
        return

    # ── Top pick ─────────────────────────────────────────────────────────────
    top = results[0]
    top_imgs = [load_pil(p) for p in top['image_paths'][:5]]
    top_imgs = [img for img in top_imgs if img]

    st.markdown('<div class="top-pick-card">', unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        if top_imgs:
            img_cols = st.columns(len(top_imgs))
            for i, img in enumerate(top_imgs):
                with img_cols[i]:
                    st.image(img, use_container_width=True)

    with right:
        st.markdown(f"""
        <div style="padding:8px 0">
          <div class="top-pick-badge">⭐ #1 BEST MATCH</div>
          <div class="top-pick-name">{top['name'] or 'Outfit'}</div>
          <div class="top-pick-desc">
            Highest cosine similarity to your style profile among all {gender_label.lower()} outfits in the database.
          </div>
          <div class="stats-row">
            <div class="stat-item">
              <div class="stat-val">{top['score']*100:.1f}%</div>
              <div class="stat-lbl">Match Score</div>
            </div>
            <div class="stat-item">
              <div class="stat-val">{len(top['image_paths'])}</div>
              <div class="stat-lbl">Items</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Results grid — 5 per row ──────────────────────────────────────────────
    st.markdown(f'<div style="font-size:16px;font-weight:700;color:#111827;margin-bottom:16px;">All 10 {gender_label} Outfit Matches</div>', unsafe_allow_html=True)

    for row_start in range(0, len(results), 5):
        row_results = results[row_start:row_start + 5]
        cols = st.columns(5)

        for col_i, result in enumerate(row_results):
            rank = row_start + col_i + 1
            imgs = [load_pil(p) for p in result['image_paths'][:4]]
            imgs = [img for img in imgs if img]

            with cols[col_i]:
                st.markdown(f"""
                <div class="outfit-card">
                  <div class="outfit-rank-bar">{'⭐ #1 BEST MATCH' if rank == 1 else f'#{rank}'}</div>
                """, unsafe_allow_html=True)

                if imgs:
                    if len(imgs) >= 2:
                        c1, c2 = st.columns(2)
                        with c1: st.image(imgs[0], use_container_width=True)
                        with c2: st.image(imgs[1], use_container_width=True)
                    if len(imgs) >= 4:
                        c1, c2 = st.columns(2)
                        with c1: st.image(imgs[2], use_container_width=True)
                        with c2: st.image(imgs[3], use_container_width=True)
                    elif len(imgs) == 3:
                        c1, c2 = st.columns(2)
                        with c1: st.image(imgs[2], use_container_width=True)

                score_pct = result['score'] * 100
                st.markdown(f"""
                  <div class="outfit-meta">
                    <div class="outfit-name">{result['name'] or 'Outfit'}</div>
                    <div class="score-row">
                      <div class="score-bar-outer">
                        <div class="score-bar-inner" style="width:{score_pct:.0f}%"></div>
                      </div>
                      <div class="score-text">{score_pct:.1f}%</div>
                    </div>
                    <div class="outfit-tags">
                      <span class="tag">{len(result['image_paths'])} items</span>
                      <span class="tag">{gender_label}</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────

if st.session_state.page == 'upload':
    page_upload()
elif st.session_state.page == 'results':
    page_results()
