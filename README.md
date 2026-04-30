# 🎵 Spotify Playlist Archetype Classifier

A big data ML pipeline that classifies songs into playlist "archetypes" (study focus, heartbreak/sad, summertime party, rustic nostalgia, jazz lounge, hype rap) using a combination of **Spotify audio features** and **NLP-derived lyric features**.

Built as a final project for CIS 2450: Big Data Analytics.

---

## Overview

Most music recommendation systems treat songs as isolated data points. This project takes a different angle: **can we predict the *vibe* a song belongs to** — not just its genre — by combining what it sounds like with what it says?

To answer that, we built a full end-to-end pipeline: from raw playlist data scraped across ~1M Spotify playlists, through lyric retrieval and NLP analysis, to a tuned LightGBM classifier that achieves **~70% accuracy across 6 balanced archetype classes**.

---

## Pipeline Architecture

```
Spotify Million Playlist Dataset (MPD)
        ↓  keyword scoring + archetype assignment
    mpd_tracks.csv  (~150k tracks/archetype)
        ↓  join on track ID / fuzzy name match
    1.2M Spotify Songs Dataset (audio features)
        ↓
    track_info.csv  (audio features per track)
        ↓  fuzzy match → Genius lyrics dataset
    Lyric NLP Analysis (VADER, j-hartmann, sentence-transformers)
        ↓
    dataset_with_lyrics_FINAL.csv  (audio + lyric features)
        ↓  imputation + encoding + train/test split
    ML-ready feature matrices (X_train, X_test, y_train, y_test)
        ↓
    Logistic Regression → Decision Tree → Random Forest → LightGBM
```

---

## Archetype Definitions

| Archetype | Description | Example Playlist Names |
|---|---|---|
| `study_focus` | Instrumental/ambient music for concentration | "lofi study beats", "deep focus coding" |
| `heartbreak_sad` | Sad, emotional, breakup themes | "crying myself to sleep", "heartbreak playlist" |
| `summertime_party` | Upbeat, summery, social | "beach party vibes", "summer bbq" |
| `rustic_nostalgia` | Country, folk, Americana | "country roads", "southern front porch" |
| `jazz_lounge` | Jazz, bossa nova, smooth/classy | "jazz cocktail hour", "lounge standards" |
| `hype_rap` | High-energy trap/hip-hop | "gym hype rap", "trap workout bangers" |

Playlists were scored using a weighted keyword system (strong: +3, medium: +2, weak: +1) and assigned to the best-matching archetype with a minimum score threshold of 3.

---

## Datasets Used

| Dataset | Source | Purpose |
|---|---|---|
| Spotify Million Playlist Dataset | [Kaggle](https://www.kaggle.com/datasets/himanshuwagh/spotify-million) | Archetype labeling via playlist names |
| Spotify 1.2M Songs (tracks_features.csv) | [Kaggle](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) | Audio features (danceability, energy, valence, etc.) |
| Genius Song Lyrics | [Kaggle](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) | Lyrics for NLP analysis |

---

## Feature Engineering

### Audio Features (14)
Spotify's precomputed audio analysis: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `time_signature`, `year`

### Lyric Features (18 numeric + 2 encoded categorical)
Derived via a multi-model NLP pipeline run on matched lyrics:

- **VADER Sentiment** — positive, negative, neutral, compound scores
- **Emotion Classification** — 7-class scores (anger, disgust, fear, joy, neutral, sadness, surprise) via [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- **Archetype Similarity** — cosine similarity between lyric embedding and each archetype's seed phrase, via `all-MiniLM-L6-v2`
- **Words Per Second** — lyric density relative to track duration
- **Dominant Emotion** (encoded), **Language** (encoded), **is_english** (binary)

### Lyric Matching
Lyrics were matched to tracks using a two-pass fuzzy matching strategy (via `rapidfuzz`):
1. Exact artist match + fuzzy title match (threshold: 85)
2. Fuzzy artist match + fuzzy title match

### Missing Value Strategy
- **Instrumental tracks** (`instrumentalness ≥ 0.5`): structural fills (sentiment → 0, neutral emotion → 1.0, etc.)
- **Unmatched non-instrumental tracks**: class-conditional mean imputation from training matched rows
- **Remaining nulls**: KNN imputation (k=5) using audio features as neighbors

---

## EDA Highlights

- **Loudness ↔ Energy** are strongly positively correlated; **Acousticness** is strongly negatively correlated with both — consistent with physical expectations
- **Hype rap** tracks have the highest words-per-second (lyric density), followed by heartbreak/sad
- **Valence (audio) vs. VADER compound (lyrics)** showed moderate positive correlation across most archetypes, confirming the two feature types are complementary rather than redundant
- **Emotion heatmap**: joy dominates summertime_party; sadness dominates heartbreak_sad; neutral dominates study_focus and jazz_lounge — validation that NLP features carry real signal

---

## Model Results

All models trained on an 80/20 stratified train/test split.

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (baseline) | 59.0% | — |
| Decision Tree (tuned) | 60.0% | — |
| Random Forest (baseline) | 68.5% | 0.681 |
| Random Forest (Optuna-tuned) | 68.6% | 0.682 |
| **LightGBM (Optuna-tuned)** | **69.8%** | **~0.695** |

### Feature Ablation (Random Forest)
| Feature Set | Accuracy |
|---|---|
| Audio only | ~63% |
| Lyric only | ~58% |
| Audio + Lyric | ~68.5% |

Both modalities contribute independently — neither dominates, and combining them provides the best result.

### Hardest Classes to Distinguish
`heartbreak_sad` and `summertime_party` showed the lowest per-class recall, likely because musical mood is highly genre-agnostic: sad songs and summer songs exist across rap, jazz, country, and pop.

---

## Tech Stack

- **Data processing**: `polars`, `pandas`, `numpy`
- **NLP**: `vaderSentiment`, `sentence-transformers` (`all-MiniLM-L6-v2`), `transformers` (`j-hartmann/emotion-english-distilroberta-base`)
- **Fuzzy matching**: `rapidfuzz`
- **ML**: `scikit-learn`, `lightgbm`, `optuna`
- **Visualization**: `matplotlib`, `seaborn`
- **Environment**: Google Colab (GPU runtime for emotion model inference)

---

## Repo Structure

```
├── cis2450_final_project.py   # Full pipeline (data collection → EDA → ML)
└── README.md
```

> **Note:** The processed CSVs (`track_info.csv`, `dataset_with_lyrics_FINAL.csv`, ML-ready matrices) are not included due to file size. Re-running the notebook from scratch requires Kaggle API credentials and a Google Drive mount for intermediate outputs. GPU runtime is recommended for the lyric NLP section.

---

## Key Takeaways

1. **Playlist name keyword scoring is a viable weak-supervision labeling strategy** at scale — no manual annotation required for 150k+ tracks per class.
2. **Audio + lyric features are genuinely complementary**: ablation confirmed each modality adds independent signal.
3. **~70% accuracy on a 6-class "vibe" classification task is meaningful** given how subjective and genre-agnostic mood categories are — especially for broad archetypes like heartbreak_sad and summertime_party.
4. **LightGBM slightly outperforms Random Forest** on this tabular feature set, consistent with the broader literature.
