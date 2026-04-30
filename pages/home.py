"""Home page"""

import dash
from dash import html, dcc
from utils import (
    ARCHETYPES, ARCHETYPE_LABELS, ARCHETYPE_EMOJI, ARCHETYPE_COLORS,
    ARCHETYPE_DESC, CLASS_COUNTS, TOTAL_TRACKS, MODEL_RESULTS,
    ALL_FEATURES, AUDIO_FEATURES, LYRIC_NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    load_dataframe,
)

dash.register_page(__name__, path="/", name="Home")


def _data_status_banner():
    """Tell the user whether they're seeing real data or demo data."""
    _, df_is_real = load_dataframe()
    if df_is_real:
        return html.Div(className="banner banner-ok", children=[
            "✓ Running on the real project dataset."
        ])
    return html.Div(className="banner banner-info", children=[
        "Demo mode: ",
        html.Code("data/tracks.csv"),
        " not found. Showing synthetic data that matches the actual class "
        "distribution and lyric statistics from the project notebook. "
        "Drop the real CSV into ",
        html.Code("data/"),
        " to swap in the live dataset.",
    ])


def _hero():
    return html.Section(className="hero", children=[
        html.H1(className="hero-title", children=[
            html.Span("Vibe", className="hero-grad-1"),
            " ",
            html.Span("Check", className="hero-grad-2"),
        ]),
        html.P(className="hero-sub", children=
            "A multi-class classifier that predicts which of six listening "
            "archetypes a song belongs to — using Spotify audio features "
            "and Genius-derived lyrical features."
        ),
        html.Div(className="hero-byline", children=[
            "Kieran Chetty & Tanner Shah · CIS 2450 Final Project · Spring 2026"
        ]),
    ])


def _stats_row():
    """Headline metrics from the actual notebook."""
    items = [
        (f"{TOTAL_TRACKS:,}",                "Tracks (post-dedup)"),
        ("6",                                "Listening archetypes"),
        (f"{len(ALL_FEATURES)}",             "Total features"),
        (f"{MODEL_RESULTS['test_acc']*100:.1f}%", "Test accuracy"),
        (f"{MODEL_RESULTS['test_macro_f1']*100:.1f}%", "Macro F1"),
        (f"{MODEL_RESULTS['cv_mean']:.3f}",  "5-fold CV F1"),
    ]
    return html.Section(className="stats-row", children=[
        html.Div(className="stat-card", children=[
            html.Div(value, className="stat-value"),
            html.Div(label, className="stat-label"),
        ]) for value, label in items
    ])


def _archetype_grid():
    cards = []
    for arch in ARCHETYPES:
        color = ARCHETYPE_COLORS[arch]
        emoji = ARCHETYPE_EMOJI[arch]
        label = ARCHETYPE_LABELS[arch]
        desc = ARCHETYPE_DESC[arch]
        count = CLASS_COUNTS[arch]
        share = count / TOTAL_TRACKS
        cards.append(
            html.Div(className="archetype-card",
                     style={"--accent": color}, children=[
                html.Div(className="archetype-card-top"),
                html.Div(className="archetype-card-body", children=[
                    html.Div(className="archetype-emoji", children=emoji),
                    html.H3(label, className="archetype-name"),
                    html.P(desc, className="archetype-desc"),
                    html.Div(className="archetype-stats", children=[
                        html.Span(f"{count:,}", className="archetype-count"),
                        html.Span(f"{share*100:.1f}% of dataset",
                                  className="archetype-share"),
                    ]),
                ]),
            ])
        )
    return html.Section(className="archetype-grid", children=[
        html.H2("The six archetypes", className="section-title"),
        html.P("Class distribution after deduplicating cross-archetype "
               "tracks (kept the row with highest archetype-keyword score).",
               className="section-sub"),
        html.Div(className="archetype-cards", children=cards),
    ])


def _project_overview():
    """How the project is structured — mirrors the notebook's pipeline."""
    steps = [
        ("1", "Source",
         "Spotify Million Playlist Dataset (1M+ playlists) + Spotify 1.2M "
         "Songs Dataset for audio features. Genius lyrics dataset "
         "(~5M songs) for text."),
        ("2", "Label",
         "Each playlist gets a per-archetype score from a strong/medium/weak "
         "keyword dictionary. Tracks inherit their playlists' archetypes "
         "weighted by score; ties broken by total_arch_score."),
        ("3", "Audio EDA",
         "Distributions, outlier checks (tempo=0, sub-minute durations), "
         "feature correlation matrix, per-archetype boxplots, "
         "Core-vs-Fringe (top-25%-score) confidence-tier visuals."),
        ("4", "Lyric NLP",
         "Fuzzy match (RapidFuzz, threshold 85) to Genius. VADER sentiment, "
         "j-hartmann emotion classifier (7-class), sentence-transformer "
         "archetype-similarity scores."),
        ("5", "ML Prep",
         "Stratified 80/20 split, structural imputation for instrumentals, "
         "class-conditional means for unmatched, KNN imputation (k=5) for "
         "remaining lyric nulls. Label-encode categoricals."),
        ("6", "Model",
         "Seven classifiers compared head-to-head — Logistic Regression, Decision "
         "Tree (baseline + GridSearch-tuned), Random Forest (baseline + Optuna-tuned), "
         "and LightGBM (baseline + Optuna-tuned). Tuned LightGBM wins at "
         f"{MODEL_RESULTS['test_macro_f1']:.3f} test macro F1. "
         "Audio + Lyric ablation confirms both modalities contribute orthogonal signal."),
    ]
    return html.Section(className="overview", children=[
        html.H2("How it works", className="section-title"),
        html.Div(className="steps", children=[
            html.Div(className="step", children=[
                html.Div(num, className="step-num"),
                html.Div(className="step-body", children=[
                    html.H4(title, className="step-title"),
                    html.P(body, className="step-text"),
                ]),
            ]) for num, title, body in steps
        ]),
    ])


def _feature_breakdown():
    return html.Section(className="feature-breakdown", children=[
        html.H2("Feature engineering", className="section-title"),
        html.P("All 37 features fed into the model, grouped by source.",
               className="section-sub"),
        html.Div(className="feature-groups", children=[
            html.Div(className="feature-group", children=[
                html.Div(className="feature-group-header", children=[
                    html.Span("Audio", className="feature-group-name"),
                    html.Span(f"{len(AUDIO_FEATURES)} features",
                              className="feature-group-count"),
                ]),
                html.Div(className="feature-tags", children=[
                    html.Span(f, className="feature-tag tag-audio")
                    for f in AUDIO_FEATURES
                ]),
            ]),
            html.Div(className="feature-group", children=[
                html.Div(className="feature-group-header", children=[
                    html.Span("Lyric (NLP-derived)", className="feature-group-name"),
                    html.Span(f"{len(LYRIC_NUMERIC_FEATURES)} features",
                              className="feature-group-count"),
                ]),
                html.Div(className="feature-tags", children=[
                    html.Span(f, className="feature-tag tag-lyric")
                    for f in LYRIC_NUMERIC_FEATURES
                ]),
            ]),
            html.Div(className="feature-group", children=[
                html.Div(className="feature-group-header", children=[
                    html.Span("Categorical (encoded)", className="feature-group-name"),
                    html.Span(f"{len(CATEGORICAL_FEATURES)} features",
                              className="feature-group-count"),
                ]),
                html.Div(className="feature-tags", children=[
                    html.Span(f, className="feature-tag tag-cat")
                    for f in CATEGORICAL_FEATURES
                ]),
            ]),
        ]),
    ])


layout = html.Div(className="page-home", children=[
    _data_status_banner(),
    _hero(),
    _stats_row(),
    _archetype_grid(),
    _project_overview(),
    _feature_breakdown(),
])
