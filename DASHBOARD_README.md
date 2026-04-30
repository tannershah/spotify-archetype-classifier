# Vibe Check Dashboard

Interactive Plotly Dash dashboard for the CIS 2450 final project — a six-archetype song classifier built on Spotify audio features and Genius-derived lyrical features.

**Built by Kieran Chetty & Tanner Shah · Spring 2026**

---

## What's in here

A multi-page Dash app with five sections:

1. **Home** — project overview, the six archetypes (`study_focus`, `jazz_lounge`, `summertime_party`, `heartbreak_sad`, `rustic_nostalgia`, `hype_rap`), real class distribution, headline metrics, and a feature-engineering breakdown of all 37 features.

2. **Pipeline** — the actual data flow: MPD scrape → audio feature join → Genius lyric retrieval → NLP feature extraction → imputation → final feature matrix → RandomForest. Mirrors the structure of the project notebook.

3. **EDA** — *the actual plots from the project notebook*, embedded directly. Tabbed into Audio EDA (correlations, distributions, per-archetype boxplots, confidence-tier visualizations) and Lyric EDA (sentiment by archetype, dominant emotion heatmap, archetype-similarity self-check, words-per-second, valence-vs-compound). Plus an interactive scatter where you can pick any two of the 32 numeric features as X/Y axes.

4. **Performance** — every real number from the notebook's evaluation: 68.5% test accuracy, 0.681 macro F1, full per-class P/R/F1, the actual confusion matrix image, the audio-vs-lyric-vs-combined ablation, 5-fold CV scores, and the real feature-importance chart.

5. **Predictor** — 14 audio-feature sliders + 6 archetype preset buttons. The 18 lyric features and 5 categorical features auto-fill to typical values for the most recently-loaded archetype preset, so you only have to think about audio. Live prediction with confidence headline and sorted probability bars.

## Setup

```bash
unzip vibe_check_dashboard.zip
cd vibe_check
pip install -r requirements.txt
python app.py
```

Opens at `http://127.0.0.1:8050`.

## Plugging in your real data (optional, for the live demo)

The dashboard runs out-of-the-box on **synthetic data** that matches the actual class distribution and lyric statistics from the project notebook. The static plots and all numbers (accuracy, F1, etc.) are real — only the live predictor uses synthesized data when you don't have the trained model.

To run it on the real model, drop two files into `data/`:

### `data/tracks.csv`
The cleaned, deduplicated dataset (32,112 rows) with these columns:

- **Identifiers**: `track_name`, `artist_name`, `archetype` (one of: `study_focus`, `jazz_lounge`, `summertime_party`, `heartbreak_sad`, `rustic_nostalgia`, `hype_rap`)
- **14 audio features**: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `time_signature`, `year`
- **18 lyric features**: `words_per_second`, `lyric_sentiment_pos`, `lyric_sentiment_neg`, `lyric_sentiment_neu`, `lyric_compound_score`, `emotion_score_anger`, `emotion_score_disgust`, `emotion_score_fear`, `emotion_score_joy`, `emotion_score_neutral`, `emotion_score_sadness`, `emotion_score_surprise`, `lyric_arch_sim_study_focus`, `lyric_arch_sim_heartbreak_sad`, `lyric_arch_sim_summertime_party`, `lyric_arch_sim_rustic_nostalgia`, `lyric_arch_sim_jazz_lounge`, `lyric_arch_sim_hype_rap`
- **5 categorical (encoded)**: `dominant_emotion_enc`, `lyrics_language_enc`, `is_english`, `explicit`, `has_lyrics`

In Colab, after your ML-prep cells:

```python
# Reconstruct the full enriched dataset with archetype labels
df_for_dashboard = df[ALL_FEATURES + ["archetype", "track_name", "artist_name"]].copy()
df_for_dashboard.to_csv("/content/drive/MyDrive/tracks.csv", index=False)
```

Then download from Drive and put it in `data/tracks.csv`.

### `data/model.pkl`
Your trained `rf` model, pickled. In Colab:

```python
import pickle
with open("/content/drive/MyDrive/model.pkl", "wb") as f:
    pickle.dump(rf, f)
```

The pickled object must:
- Have `.predict()`, `.predict_proba()`, and `.classes_` attributes
- Expect feature order: `AUDIO_FEATURES + LYRIC_NUMERIC_FEATURES + CATEGORICAL_FEATURES` (37 total) — same order as in the notebook

## File layout

```
vibe_check/
├── app.py                      # Dash entry, navbar, routing
├── utils.py                    # Constants, schema, archetype profiles, predictor
├── requirements.txt
├── README.md
├── pages/
│   ├── home.py                 # Hero + archetype cards + project overview
│   ├── pipeline.py             # End-to-end pipeline visualization
│   ├── eda.py                  # Notebook plots in tabs + interactive scatter
│   ├── performance.py          # Real model evaluation
│   └── predictor.py            # 14 audio sliders + 6 presets
├── assets/
│   ├── style.css               # Editorial/magazine theme
│   └── notebook_plots/         # 22 plots extracted from your notebook
└── data/                       # Drop tracks.csv and model.pkl here
```

## What's grounded in the notebook vs. synthesized

**Real (extracted directly from the notebook):**
- Six archetype names, class distribution (32,112 rows post-dedup)
- 37-feature schema (14 audio + 18 lyric + 5 categorical)
- All performance numbers (test acc 68.53%, macro F1 0.6810, CV 0.6729 ± 0.0268)
- Per-class precision/recall/F1
- Ablation results (audio 0.5734 / lyric 0.5861 / combined 0.6801)
- All 22 EDA plots (embedded as PNGs in `assets/notebook_plots/`)
- Per-archetype lyric feature means (from notebook cell 67)

**Synthesized for the demo (when real files absent):**
- The 32,112-row interactive scatter dataset (uses archetype profiles + Gaussian jitter)
- The live predictor's RandomForest (trained on the synthetic data, in the same place a real `model.pkl` would slot in)

Drop in your real `tracks.csv` and `model.pkl` and the synthetic fallbacks vanish — every prediction becomes a real model call against your actual training data.

## Demo tips for the 8-10 min recording

Suggested narrative arc:

1. **Home** (1 min) — open with hero, describe the problem ("can we predict listening archetype from audio + lyrics?"), point at the six archetypes and the dataset size.
2. **Pipeline** (2 min) — walk down the eight stages. The keyword-dictionary labeling and the three-population imputation strategy are good "design decision" talking points.
3. **EDA** (2 min) — switch between Audio and Lyric tabs. The correlation matrix and the per-archetype boxplots show why the model can separate classes; the dominant-emotion heatmap and archetype-similarity self-check are the lyric-NLP money shots.
4. **Performance** (2 min) — headline metrics, then the confusion matrix (point at the diagonal), then the **ablation chart** (this is your strongest slide — audio alone 0.573, lyric alone 0.586, combined 0.680, proving both modalities pull weight), then feature importance.
5. **Predictor** (2 min) — load the Hype/Rap preset, point at the prediction (~50%+ confidence), then drag energy down and watch it flip toward Heartbreak/Sad. This is the interactive payoff.

Don't forget the AI-attribution slide at the end — the rubric requires it.

## AI-usage attribution

Dashboard scaffolding (Dash app structure, page routing, CSS theme, callback patterns, slider mechanics) was built with Anthropic's Claude. All schema, archetype names, model results, and EDA plots come directly from the project notebook authored by Kieran Chetty and Tanner Shah. The synthetic-data fallback and the auto-fill predictor logic were also generated with Claude assistance.
