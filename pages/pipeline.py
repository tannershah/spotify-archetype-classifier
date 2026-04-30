"""Pipeline page — visualizes the actual end-to-end data flow."""

import dash
from dash import html
from utils import CLASS_COUNTS, TOTAL_TRACKS

dash.register_page(__name__, path="/pipeline", name="Pipeline")


def _stage(num, title, sub, items, color):
    return html.Div(className="pipe-stage", style={"--accent": color}, children=[
        html.Div(className="pipe-stage-head", children=[
            html.Span(num, className="pipe-num"),
            html.Div(className="pipe-stage-titles", children=[
                html.H3(title, className="pipe-stage-title"),
                html.P(sub, className="pipe-stage-sub"),
            ]),
        ]),
        html.Ul(className="pipe-bullets", children=[
            html.Li(item) for item in items
        ]),
    ])


def _arrow():
    return html.Div(className="pipe-arrow", children="↓")


layout = html.Div(className="page-pipeline", children=[
    html.Section(className="page-header", children=[
        html.H1("Data Pipeline", className="page-title"),
        html.P("From raw playlist scrape to a trained 6-class classifier — "
               "every stage that produced the final 32,112-row labeled dataset.",
               className="page-sub"),
    ]),

    html.Div(className="pipeline-flow", children=[
        _stage("1", "Archetype keyword dictionary",
               "Hand-curated 3-tier keyword lists per archetype",
               [
                   "Strong (+3), medium (+2), weak (+1) keywords for each of 6 archetypes",
                   "Exclusions for false-positive playlist names "
                   "('not country', 'therapy', 'crystal', etc.)",
                   "Designed to balance precision and recall on playlist-name signals",
               ],
               "#A78BFA"),
        _arrow(),

        _stage("2", "MPD playlist scrape",
               "Spotify Million Playlist Dataset (5.2 GB)",
               [
                   "Stream all 1M playlists, score each against keyword dictionary",
                   "For each (playlist, archetype) above threshold, pull track IDs",
                   "Cap at 150,000 tracks per archetype for balance",
                   "Output: mpd_tracks.csv with playlist-derived archetype labels",
               ],
               "#5B8DEF"),
        _arrow(),

        _stage("3", "Audio feature join",
               "Spotify 1.2M Songs Dataset",
               [
                   "Pass 1: exact track-ID match (25,618 rows)",
                   "Pass 2: name + artist match for unmatched (30,741 rows)",
                   "Combined: 45,164 rows × 24 cols (with cross-archetype duplicates)",
                   "32,112 unique tracks once deduplicated",
               ],
               "#60A5FA"),
        _arrow(),

        _stage("4", "Lyric retrieval",
               "Genius Song Lyrics dataset (Kaggle)",
               [
                   "Stream Kaggle CSV in 100k-row chunks, filter by artist tokens",
                   "RapidFuzz matching: exact-artist → fuzzy-title (≥85), then fuzzy-artist fallback",
                   "Normalize titles (strip 'feat.', remasters, parens)",
                   "Final match rate: 59.1% — 18,973 tracks with lyrics",
               ],
               "#FBBF24"),
        _arrow(),

        _stage("5", "Lyric feature extraction",
               "Three NLP models in batches of 5,000",
               [
                   "VADER → 4 sentiment scores (pos/neg/neu/compound)",
                   "j-hartmann/emotion-english-distilroberta-base → 7 emotion scores",
                   "all-MiniLM-L6-v2 sentence-transformer → 6 archetype-similarity scores",
                   "Plus words_per_second from raw text + duration",
                   "Output: 18 numeric lyric features per track",
               ],
               "#EF4444"),
        _arrow(),

        _stage("6", "Imputation strategy",
               "Three populations, three strategies",
               [
                   "Instrumentals (instrumentalness ≥ 0.5): structural zeros + lyric flags off",
                   "Unmatched-but-likely-has-lyrics: class-conditional means from train-matched only",
                   "Remaining lyric nulls: KNN (k=5) using audio features as basis",
                   "Fitted on TRAIN only → applied to TEST (no leakage)",
               ],
               "#D97706"),
        _arrow(),

        _stage("7", "Final feature matrix",
               "37 features, stratified 80/20 split",
               [
                   "14 audio features (Spotify)",
                   "18 lyric numeric features (NLP)",
                   "5 categorical encodings (dominant_emotion, language, is_english, explicit, has_lyrics)",
                   f"Train: 25,689 rows · Test: 6,423 rows",
               ],
               "#A78BFA"),
        _arrow(),

        _stage("8", "Random Forest classifier",
               "300 trees, balanced class weights, 5-fold CV",
               [
                   "RandomForestClassifier(n=300, min_samples_leaf=2, class_weight='balanced')",
                   "Test accuracy: 68.5% · Macro F1: 0.681",
                   "5-fold CV F1: 0.673 ± 0.027 (low variance — stable model)",
                   "Audio-only F1 = 0.573, lyric-only = 0.586, combined = 0.680 → both modalities matter",
               ],
               "#5B8DEF"),
    ]),

    html.Section(className="pipeline-takeaways", children=[
        html.H2("Key design decisions", className="section-title"),
        html.Div(className="takeaway-grid", children=[
            html.Div(className="takeaway", children=[
                html.H4("Why playlist-name labeling?"),
                html.P("Hand-labeling 32k songs is infeasible. Playlist names "
                       "are noisy but at scale they form a strong signal — "
                       "users curate playlists with semantic intent."),
            ]),
            html.Div(className="takeaway", children=[
                html.H4("Why fuzzy matching?"),
                html.P("Track titles vary across catalogs ('feat. X' suffixes, "
                       "remaster years, alt punctuation). Exact match would "
                       "throw away ~20% of recoverable rows."),
            ]),
            html.Div(className="takeaway", children=[
                html.H4("Why three imputation regimes?"),
                html.P("Lyric-null reasons differ: instrumentals genuinely have "
                       "no lyrics, missing matches likely do. Treating them the "
                       "same would mis-impute both groups."),
            ]),
            html.Div(className="takeaway", children=[
                html.H4("Why the audio + lyric ablation?"),
                html.P("Confirms both modalities contribute non-trivially. "
                       "Audio alone: 57% F1. Lyrics alone: 59% F1. Combined: "
                       "68% F1 — synergy worth the engineering."),
            ]),
        ]),
    ]),
])
