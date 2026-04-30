""" H2H comparison of all seven models trained
in the project notebook plus the deep-dive metrics for any selected model."""

import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go

from utils import (
    ARCHETYPES, ARCHETYPE_LABELS, ARCHETYPE_COLORS,
    MODEL_RESULTS, MODELS, MODEL_ORDER, BEST_MODEL_KEY,
    N_TRAIN, N_TEST,
)

dash.register_page(__name__, path="/performance", name="Performance")


def _metric_card(value, label, color="#A78BFA"):
    return html.Div(className="metric-card", style={"--accent": color}, children=[
        html.Div(value, className="metric-value"),
        html.Div(label, className="metric-label"),
    ])


def _fmt_pct(v):
    return f"{v*100:.1f}%" if v is not None else "—"


def _fmt_f1(v):
    return f"{v:.3f}" if v is not None else "—"



def _headline_metrics():
    best = MODELS[BEST_MODEL_KEY]
    return html.Section(className="metric-grid", children=[
        _metric_card(_fmt_pct(best["test_acc"]),    "Test accuracy",    "#A78BFA"),
        _metric_card(_fmt_f1(best["test_macro_f1"]), "Test macro F1",   "#5B8DEF"),
        _metric_card(_fmt_f1(best["cv_macro_f1"]),  "CV macro F1 (3-fold)", "#FBBF24"),
        _metric_card(f"{N_TEST:,}",                  "Test set size",   "#60A5FA"),
        _metric_card(_fmt_pct(best["train_acc"]),    "Train accuracy",  "#EF4444"),
        _metric_card(best["name"],                   "Best model",      "#D97706"),
    ])



def _models_comparison_chart():
    """Grouped bar of test-accuracy and test-macro-F1 across all 7 models."""
    keys   = MODEL_ORDER
    names  = [MODELS[k]["name"] for k in keys]
    accs   = [MODELS[k]["test_acc"]      for k in keys]
    f1s    = [MODELS[k]["test_macro_f1"] for k in keys]

    bar_colors_acc = [
        "#EF4444" if k == BEST_MODEL_KEY else "#5B8DEF" for k in keys
    ]
    bar_colors_f1  = [
        "#F59E0B" if k == BEST_MODEL_KEY else "#A78BFA" for k in keys
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Test accuracy", x=names, y=accs,
        marker_color=bar_colors_acc,
        text=[f"{v:.3f}" for v in accs], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Test macro F1", x=names, y=f1s,
        marker_color=bar_colors_f1,
        text=[f"{v:.3f}" for v in f1s], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", barmode="group", height=460,
        paper_bgcolor="#0F0F14", plot_bgcolor="#0F0F14",
        margin=dict(l=60, r=20, t=20, b=120),
        yaxis=dict(range=[0.4, 0.78], gridcolor="#2A2A33", title="Score"),
        xaxis=dict(tickangle=-25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#E5E7EB"),
    )
    return fig


def _models_summary_table():
    """Compact table — one row per model — with all the headline numbers."""
    rows = [html.Tr([
        html.Th("Model"),
        html.Th("Family"),
        html.Th("Train acc"),
        html.Th("Test acc"),
        html.Th("Test macro F1"),
        html.Th("Tuning"),
    ])]
    for k in MODEL_ORDER:
        m = MODELS[k]
        is_best = k == BEST_MODEL_KEY
        name_cell = html.Td(children=[
            html.Strong(m["name"]),
            html.Span(" · BEST", className="best-badge") if is_best else None,
        ])
        rows.append(html.Tr(
            className="row-best" if is_best else "",
            children=[
                name_cell,
                html.Td(m["family"]),
                html.Td(_fmt_pct(m["train_acc"])),
                html.Td(_fmt_pct(m["test_acc"])),
                html.Td(_fmt_f1(m["test_macro_f1"])),
                html.Td(m.get("tuner", "—")),
            ],
        ))
    return html.Table(className="report-table", children=rows)


def _model_picker():
    return html.Div(className="model-picker", children=[
        html.Label("Inspect model:", htmlFor="model-select"),
        dcc.Dropdown(
            id="model-select",
            options=[{"label": MODELS[k]["name"], "value": k} for k in MODEL_ORDER],
            value=BEST_MODEL_KEY,
            clearable=False,
            style={"minWidth": "260px"},
        ),
    ])


def _per_class_table_for(model_key):
    m = MODELS[model_key]
    rows = [html.Tr([
        html.Th("Archetype"), html.Th("Precision"),
        html.Th("Recall"), html.Th("F1"), html.Th("Support"),
    ])]
    for arch in sorted(ARCHETYPES):
        pc = m["per_class"][arch]
        rows.append(html.Tr([
            html.Td(html.Span(ARCHETYPE_LABELS[arch],
                              className="arch-pill",
                              style={"--accent": ARCHETYPE_COLORS[arch]})),
            html.Td(f"{pc['precision']:.2f}"),
            html.Td(f"{pc['recall']:.2f}"),
            html.Td(f"{pc['f1']:.2f}"),
            html.Td(f"{pc['support']:,}"),
        ]))

    avg_p = sum(m["per_class"][a]["precision"] for a in ARCHETYPES) / 6
    avg_r = sum(m["per_class"][a]["recall"]    for a in ARCHETYPES) / 6
    avg_f = sum(m["per_class"][a]["f1"]        for a in ARCHETYPES) / 6
    rows.append(html.Tr(className="row-total", children=[
        html.Td(html.Strong("Macro avg")),
        html.Td(f"{avg_p:.2f}"),
        html.Td(f"{avg_r:.2f}"),
        html.Td(f"{avg_f:.2f}"),
        html.Td(f"{N_TEST:,}"),
    ]))
    return html.Table(className="report-table", children=rows)


def _per_class_chart_for(model_key):
    m = MODELS[model_key]
    classes = sorted(ARCHETYPES)
    precisions = [m["per_class"][c]["precision"] for c in classes]
    recalls    = [m["per_class"][c]["recall"]    for c in classes]
    f1s        = [m["per_class"][c]["f1"]        for c in classes]
    labels     = [ARCHETYPE_LABELS[c] for c in classes]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=labels, y=precisions,
                         marker_color="#5B8DEF",
                         text=[f"{v:.2f}" for v in precisions], textposition="outside"))
    fig.add_trace(go.Bar(name="Recall",    x=labels, y=recalls,
                         marker_color="#A78BFA",
                         text=[f"{v:.2f}" for v in recalls], textposition="outside"))
    fig.add_trace(go.Bar(name="F1",        x=labels, y=f1s,
                         marker_color="#FBBF24",
                         text=[f"{v:.2f}" for v in f1s], textposition="outside"))
    fig.update_layout(
        template="plotly_dark", barmode="group", height=420,
        paper_bgcolor="#0F0F14", plot_bgcolor="#0F0F14",
        margin=dict(l=60, r=20, t=20, b=60),
        yaxis=dict(range=[0, 1], gridcolor="#2A2A33"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#E5E7EB"),
    )
    return fig


def _model_config_card(model_key):
    m = MODELS[model_key]
    rows = [
        ("Model",           m["name"]),
        ("Family",          m["family"]),
        ("Configuration",   m["config"]),
        ("Train accuracy",  _fmt_pct(m["train_acc"])),
        ("Test accuracy",   _fmt_pct(m["test_acc"])),
        ("Test macro F1",   _fmt_f1(m["test_macro_f1"])),
    ]
    if m.get("cv_macro_f1") is not None:
        rows.append(("CV macro F1", _fmt_f1(m["cv_macro_f1"])))
    if m.get("tuner"):
        rows.append(("Tuning", m["tuner"]))
    return html.Div(className="config-table", children=[
        html.Div(className="config-row", children=[
            html.Div(label, className="config-label"),
            html.Div(html.Code(value) if label in ("Configuration",) else value,
                     className="config-value"),
        ]) for label, value in rows
    ])


def _ablation_chart():
    abl = MODEL_RESULTS["ablation"]
    names = list(abl.keys())
    accs  = [abl[n]["acc"] for n in names]
    f1s   = [abl[n]["f1"]  for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=names, y=accs,
                         marker_color="#5B8DEF",
                         text=[f"{v:.3f}" for v in accs], textposition="outside"))
    fig.add_trace(go.Bar(name="Macro F1", x=names, y=f1s,
                         marker_color="#A78BFA",
                         text=[f"{v:.3f}" for v in f1s], textposition="outside"))
    fig.update_layout(
        template="plotly_dark", barmode="group", height=380,
        paper_bgcolor="#0F0F14", plot_bgcolor="#0F0F14",
        margin=dict(l=60, r=20, t=20, b=60),
        yaxis=dict(range=[0, 0.85], gridcolor="#2A2A33"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#E5E7EB"),
    )
    return fig


def _cv_chart():
    scores = MODEL_RESULTS["cv_scores"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(scores))],
        y=scores,
        marker_color="#FBBF24",
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
    ))
    fig.add_hline(y=MODEL_RESULTS["cv_mean"], line_dash="dash",
                  line_color="#A78BFA",
                  annotation_text=f"Mean: {MODEL_RESULTS['cv_mean']:.3f}",
                  annotation_position="top right")
    fig.update_layout(
        template="plotly_dark", height=320,
        paper_bgcolor="#0F0F14", plot_bgcolor="#0F0F14",
        margin=dict(l=60, r=20, t=40, b=40),
        yaxis=dict(range=[0.6, 0.75], gridcolor="#2A2A33", title="Macro F1"),
        showlegend=False,
        font=dict(color="#E5E7EB"),
    )
    return fig



layout = html.Div(className="page-performance", children=[
    html.Section(className="page-header", children=[
        html.H1("Model Performance", className="page-title"),
        html.P("All numbers below are the actual results from the project notebook. "
               "Seven models were trained on the same 25,689-row training set "
               "and evaluated on a held-out 6,423-row stratified test set: "
               "Logistic Regression, Decision Tree (baseline + GridSearch-tuned), "
               "Random Forest (baseline + Optuna-tuned), and LightGBM "
               "(baseline + Optuna-tuned). Tuned LightGBM is the winner.",
               className="page-sub"),
    ]),

    html.H2("Headline metrics — Best model (Tuned LightGBM)", className="section-title"),
    _headline_metrics(),

    html.Div(className="card", children=[
        html.H2("All seven models, head-to-head", className="section-title"),
        html.P("Test accuracy and macro-F1 for every model trained in the notebook. "
               "Highlights: Logistic Regression sets the linear floor (~0.59 F1); "
               "ensembles add a clear 0.10 lift; Tuned LightGBM edges out Tuned "
               "Random Forest by +0.013 F1 and is the production pick.",
               className="card-sub"),
        dcc.Graph(figure=_models_comparison_chart(), config={"displaylogo": False}),
        _models_summary_table(),
    ]),

    html.Div(className="card", children=[
        html.H2("Inspect a model", className="section-title"),
        html.P("Pick any of the seven models to see its configuration and "
               "per-archetype precision / recall / F1 breakdown.",
               className="card-sub"),
        _model_picker(),
        html.Div(id="model-detail"),
    ]),

    html.Div(className="card", children=[
        html.H2("Confusion matrix (Random Forest baseline, test set)", className="section-title"),
        html.P("Saved from the notebook's RF baseline. Rows are true labels, columns "
               "are predicted. Right panel shows row-normalized recall — the "
               "diagonal is the per-class recall. The tuned LightGBM model has "
               "very similar confusion structure (mostly tighter diagonals).",
               className="card-sub"),
        html.Img(src="/assets/notebook_plots/confusion_matrix.png",
                 className="notebook-plot"),
        html.Div(className="cm-insights", children=[
            html.H4("Where the model confuses things:"),
            html.Ul(children=[
                html.Li("heartbreak_sad ↔ summertime_party: 14% of heartbreak gets "
                        "called party — both are vocal pop in a similar tempo range."),
                html.Li("rustic_nostalgia ↔ heartbreak_sad: 15% bleed — country "
                        "themes overlap heavily with sad themes."),
                html.Li("study_focus ↔ heartbreak_sad: 10% bleed — both lean "
                        "low-energy, mid-acousticness."),
                html.Li("hype_rap is almost a closed cluster — only 4% bleed "
                        "into heartbreak_sad."),
            ]),
        ]),
    ]),

    html.Div(className="card", children=[
        html.H2("Feature ablation: audio vs lyric vs combined", className="section-title"),
        html.P("The headline modeling result. Audio features alone get to 0.573 F1; "
               "lyric features alone get to 0.581 F1; combining them jumps to 0.677 F1 — "
               "a +0.10 lift. Both modalities contribute genuinely orthogonal signal. "
               "(Run with the same RF used as the ablation reference.)",
               className="card-sub"),
        dcc.Graph(figure=_ablation_chart(), config={"displaylogo": False}),
    ]),

    html.Div(className="card", children=[
        html.H2("5-fold cross-validation (Random Forest baseline)", className="section-title"),
        html.P(f"Per-fold macro F1 scores. Mean = {MODEL_RESULTS['cv_mean']:.4f}, "
               f"std = {MODEL_RESULTS['cv_std']:.4f}. Low variance suggests the "
               "model generalizes consistently across data subsets — not a fluke "
               "of one lucky split.",
               className="card-sub"),
        dcc.Graph(figure=_cv_chart(), config={"displaylogo": False}),
    ]),

    html.Div(className="card", children=[
        html.H2("Hyperparameter optimization (Optuna)", className="section-title"),
        html.P("Both ensembles got a Bayesian hyperparameter search using Optuna's "
               "TPE sampler with 3-fold stratified CV on macro-F1. The Random Forest "
               "search ran for 60 trials over 7 hyperparameters; the LightGBM search "
               "ran for 40 trials over 9 hyperparameters.",
               className="card-sub"),
        html.Div(className="optuna-grid", children=[
            html.Div(className="optuna-card", children=[
                html.H4("Random Forest (Optuna best)"),
                html.P(f"CV macro F1: {MODELS['rf_tuned']['cv_macro_f1']:.4f}  →  "
                       f"Test macro F1: {MODELS['rf_tuned']['test_macro_f1']:.4f}",
                       className="optuna-headline"),
                html.Pre(MODELS["rf_tuned"]["config"], className="config-block"),
                html.P("Tuning added almost nothing on the test set "
                       "(+0.0004 F1 over baseline) — RF was already saturated.",
                       className="optuna-note"),
            ]),
            html.Div(className="optuna-card", children=[
                html.H4("LightGBM (Optuna best) — winner"),
                html.P(f"CV macro F1: {MODELS['lgbm_tuned']['cv_macro_f1']:.4f}  →  "
                       f"Test macro F1: {MODELS['lgbm_tuned']['test_macro_f1']:.4f}",
                       className="optuna-headline"),
                html.Pre(MODELS["lgbm_tuned"]["config"], className="config-block"),
                html.P("Tuning lifted LightGBM by +0.0058 F1 over its already-strong "
                       "baseline — and put it +0.013 F1 ahead of tuned RF, the second-best.",
                       className="optuna-note"),
            ]),
        ]),
    ]),

    html.Div(className="card", children=[
        html.H2("Feature importance (Random Forest)", className="section-title"),
        html.P("Mean decrease in impurity across all 300 trees. Top of the list: "
               "words_per_second is the single most important feature (clear hype_rap "
               "signal), followed by lyric_arch_sim_hype_rap, speechiness, and "
               "emotion_score_joy. Note that lyric features dominate the top half — "
               "the NLP pipeline pulled real weight.",
               className="card-sub"),
        html.Img(src="/assets/notebook_plots/feature_importance.png",
                 className="notebook-plot"),
    ]),
])


@callback(
    Output("model-detail", "children"),
    Input("model-select", "value"),
)
def _render_model_detail(model_key):
    if not model_key or model_key not in MODELS:
        model_key = BEST_MODEL_KEY
    return [
        _model_config_card(model_key),
        _per_class_table_for(model_key),
        dcc.Graph(figure=_per_class_chart_for(model_key),
                  config={"displaylogo": False}),
    ]
