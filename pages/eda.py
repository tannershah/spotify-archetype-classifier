"""EDA page — displays the actual plots produced in the project notebook,
plus an interactive scatter for free-form exploration."""

import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils import (
    ARCHETYPES, ARCHETYPE_LABELS, ARCHETYPE_COLORS,
    AUDIO_FEATURES, LYRIC_NUMERIC_FEATURES,
    load_dataframe,
)

dash.register_page(__name__, path="/eda", name="EDA")

# Plots extracted from the notebook (assets/notebook_plots/)
NOTEBOOK_PLOT = "/assets/notebook_plots/{}.png"


def _plot_panel(title, fname, sub=""):
    return html.Div(className="plot-panel", children=[
        html.H4(title, className="plot-panel-title"),
        html.P(sub, className="plot-panel-sub") if sub else None,
        html.Img(src=NOTEBOOK_PLOT.format(fname),
                 className="notebook-plot"),
    ])


def _audio_eda_section():
    return html.Div(className="eda-section", children=[
        html.H3("Audio features", className="eda-section-title"),
        html.P("Spotify-derived audio characteristics across the labeled dataset.",
               className="eda-section-sub"),

        _plot_panel(
            "Tempo and duration distributions",
            "tempo_duration_distributions",
            "Sanity check after merging — flagged 0-BPM API failures and outlier durations."
        ),
        _plot_panel(
            "Audio feature correlations",
            "audio_correlation_matrix",
            "Loudness ↔ energy strongly positive (expected). Acousticness ↔ energy "
            "strongly negative. Valence correlates with both danceability and energy."
        ),
        _plot_panel(
            "Continuous audio features by archetype",
            "audio_features_by_archetype_boxplots",
            "Boxplots across all six archetypes — speechiness clearly separates "
            "hype_rap; instrumentalness separates study_focus and jazz_lounge."
        ),
        _plot_panel(
            "Same view, second batch",
            "audio_features_by_archetype_boxplots_1",
            "Continuation of the per-feature breakdown."
        ),
    ])


def _lyric_eda_section():
    return html.Div(className="eda-section", children=[
        html.H3("Lyric features (NLP)", className="eda-section-title"),
        html.P("Features derived from Genius lyrics: VADER sentiment, j-hartmann "
               "emotion classifier, and sentence-transformer archetype-similarity scores.",
               className="eda-section-sub"),

        _plot_panel(
            "Lyric coverage by archetype",
            "lyric_match_status_by_archetype",
            "Which archetypes have lyrics, are instrumentals, or just failed to match. "
            "study_focus and jazz_lounge are heavily instrumental; hype_rap is nearly all matched."
        ),
        _plot_panel(
            "Sentiment distributions",
            "lyric_sentiment_by_archetype",
            "VADER compound score by archetype. Hype_rap skews negative; "
            "jazz_lounge and rustic_nostalgia skew positive."
        ),
        _plot_panel(
            "Dominant emotion heatmap",
            "dominant_emotion_heatmap",
            "Proportion of each archetype's matched English tracks classified into "
            "each of seven emotions by the j-hartmann model."
        ),
        _plot_panel(
            "Archetype-similarity self-check",
            "lyric_archetype_similarity",
            "Mean sentence-transformer similarity between each track's lyrics and "
            "an archetype seed phrase. heartbreak_sad and hype_rap recover their own "
            "archetype as top match; jazz_lounge and rustic_nostalgia don't (a known weakness)."
        ),
        _plot_panel(
            "Words per second by archetype",
            "words_per_second_by_archetype",
            "Lyric-density signal — hype_rap clearly stands out near 2.5 wps "
            "vs ~1.0 for everyone else."
        ),
        _plot_panel(
            "Audio profile by lyric-match status",
            "audio_features_by_lyric_status",
            "Validates the imputation strategy — instrumentals genuinely look "
            "different in audio space (lower energy, higher acousticness)."
        ),
    ])


def _interactive_scatter():
    """Free-form scatter the user can configure."""
    feat_options = [{"label": f, "value": f}
                    for f in AUDIO_FEATURES + LYRIC_NUMERIC_FEATURES]
    return html.Div(className="eda-section", children=[
        html.H3("Interactive feature explorer", className="eda-section-title"),
        html.P("Pick any two features and see how the six archetypes separate. "
               "Sampled at 4,000 rows for performance.",
               className="eda-section-sub"),

        html.Div(className="scatter-controls", children=[
            html.Div(className="control", children=[
                html.Label("X axis"),
                dcc.Dropdown(id="eda-x", options=feat_options,
                             value="energy", clearable=False),
            ]),
            html.Div(className="control", children=[
                html.Label("Y axis"),
                dcc.Dropdown(id="eda-y", options=feat_options,
                             value="valence", clearable=False),
            ]),
        ]),
        dcc.Graph(id="eda-scatter", config={"displaylogo": False}),
    ])


layout = html.Div(className="page-eda", children=[
    html.Section(className="page-header", children=[
        html.H1("Exploratory Data Analysis", className="page-title"),
        html.P("All visualizations below are the actual plots produced in the "
               "project's Colab notebook — extracted directly so you're seeing "
               "the same charts that informed the modeling decisions.",
               className="page-sub"),
    ]),

    html.Div(className="eda-tabs", children=[
        dcc.Tabs(id="eda-tabs", value="audio", className="custom-tabs", children=[
            dcc.Tab(label="Audio features", value="audio",
                    className="custom-tab", selected_className="custom-tab-active"),
            dcc.Tab(label="Lyric features", value="lyric",
                    className="custom-tab", selected_className="custom-tab-active"),
            dcc.Tab(label="Interactive", value="interactive",
                    className="custom-tab", selected_className="custom-tab-active"),
        ]),
    ]),

    html.Div(id="eda-tab-content", className="eda-tab-content"),
])


@callback(
    Output("eda-tab-content", "children"),
    Input("eda-tabs", "value"),
)
def _render_tab(tab):
    if tab == "audio":
        return _audio_eda_section()
    if tab == "lyric":
        return _lyric_eda_section()
    return _interactive_scatter()


@callback(
    Output("eda-scatter", "figure"),
    Input("eda-x", "value"),
    Input("eda-y", "value"),
)
def _scatter(x, y):
    df, _ = load_dataframe()
    sample = df.sample(n=min(4000, len(df)), random_state=42)
    fig = go.Figure()
    for arch in ARCHETYPES:
        sub = sample[sample["archetype"] == arch]
        fig.add_trace(go.Scattergl(
            x=sub[x], y=sub[y], mode="markers",
            name=ARCHETYPE_LABELS[arch],
            marker=dict(size=5, color=ARCHETYPE_COLORS[arch], opacity=0.5),
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F0F14", plot_bgcolor="#0F0F14",
        height=560, margin=dict(l=60, r=20, t=40, b=60),
        xaxis=dict(title=x, gridcolor="#2A2A33", zeroline=False),
        yaxis=dict(title=y, gridcolor="#2A2A33", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        font=dict(color="#E5E7EB"),
    )
    return fig
