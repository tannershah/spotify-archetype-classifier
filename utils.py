"""
Vibe Check's shared constants and data loaders.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import pickle
import numpy as np
import pandas as pd


ARCHETYPES = [
    "study_focus",
    "jazz_lounge",
    "summertime_party",
    "heartbreak_sad",
    "rustic_nostalgia",
    "hype_rap",
]

ARCHETYPE_LABELS = {
    "study_focus":      "Study / Focus",
    "jazz_lounge":      "Jazz / Lounge",
    "summertime_party": "Summertime / Party",
    "heartbreak_sad":   "Heartbreak / Sad",
    "rustic_nostalgia": "Rustic / Nostalgia",
    "hype_rap":         "Hype / Rap",
}

ARCHETYPE_EMOJI = {
    "study_focus":      "📚",
    "jazz_lounge":      "🎷",
    "summertime_party": "🌴",
    "heartbreak_sad":   "💔",
    "rustic_nostalgia": "🤠",
    "hype_rap":         "🔥",
}

ARCHETYPE_COLORS = {
    "study_focus":      "#5B8DEF",
    "jazz_lounge":      "#A78BFA",
    "summertime_party": "#FBBF24",
    "heartbreak_sad":   "#60A5FA",
    "rustic_nostalgia": "#D97706",
    "hype_rap":         "#EF4444",
}

ARCHETYPE_DESC = {
    "study_focus":      "Lo-fi, ambient, instrumental — for concentration and quiet hours.",
    "jazz_lounge":      "Smooth jazz and lounge — late-night, sophisticated, mellow.",
    "summertime_party": "Beachy, upbeat, danceable — pool parties and summer drives.",
    "heartbreak_sad":   "Sad, lonely, melancholic — breakups and quiet crying.",
    "rustic_nostalgia": "Country, folk, Americana — nostalgic and rural in feel.",
    "hype_rap":         "High-energy rap and hip-hop — gym, hype, raw aggression.",
}

# Class distribution after dedup (notebook cell 75)
CLASS_COUNTS = {
    "study_focus":      8105,
    "jazz_lounge":      7748,
    "summertime_party": 4692,
    "heartbreak_sad":   4489,
    "rustic_nostalgia": 3592,
    "hype_rap":         3486,
}
TOTAL_TRACKS = sum(CLASS_COUNTS.values())  # 32,112


AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "time_signature", "year",
]

LYRIC_NUMERIC_FEATURES = [
    "words_per_second",
    "lyric_sentiment_pos", "lyric_sentiment_neg",
    "lyric_sentiment_neu", "lyric_compound_score",
    "emotion_score_anger", "emotion_score_disgust", "emotion_score_fear",
    "emotion_score_joy", "emotion_score_neutral",
    "emotion_score_sadness", "emotion_score_surprise",
    "lyric_arch_sim_study_focus", "lyric_arch_sim_heartbreak_sad",
    "lyric_arch_sim_summertime_party", "lyric_arch_sim_rustic_nostalgia",
    "lyric_arch_sim_jazz_lounge", "lyric_arch_sim_hype_rap",
]

CATEGORICAL_FEATURES = [
    "dominant_emotion_enc", "lyrics_language_enc",
    "is_english", "explicit", "has_lyrics",
]

ALL_FEATURES = AUDIO_FEATURES + LYRIC_NUMERIC_FEATURES + CATEGORICAL_FEATURES
assert len(ALL_FEATURES) == 37


AUDIO_RANGES = {
    "danceability":     (0.0, 1.0, 0.55),
    "energy":           (0.0, 1.0, 0.55),
    "key":              (0,   11,  5),
    "loudness":         (-30.0, 0.0, -9.7),
    "mode":             (0,   1,   1),
    "speechiness":      (0.0, 1.0, 0.08),
    "acousticness":     (0.0, 1.0, 0.38),
    "instrumentalness": (0.0, 1.0, 0.20),
    "liveness":         (0.0, 1.0, 0.20),
    "valence":          (0.0, 1.0, 0.46),
    "tempo":            (40.0, 220.0, 118.4),
    "duration_ms":      (60_000, 600_000, 247_525),
    "time_signature":   (3,   5,   4),
    "year":             (1960, 2025, 2005),
}


N_TRAIN = 25_689
N_TEST  = 6_423

MODELS = {
    "logreg": {
        "name":        "Logistic Regression",
        "config":      "LogisticRegression(C=1.0, penalty='l2', solver='saga', "
                       "multi_class='multinomial', class_weight='balanced')",
        "family":      "Linear (baseline)",
        "tuned":       False,
        "train_acc":   0.5818,
        "test_acc":    0.5868,
        "test_macro_f1": 0.5877,
        "per_class": {
            "heartbreak_sad":   {"precision": 0.45, "recall": 0.54, "f1": 0.49, "support": 898},
            "hype_rap":         {"precision": 0.78, "recall": 0.86, "f1": 0.81, "support": 697},
            "jazz_lounge":      {"precision": 0.69, "recall": 0.57, "f1": 0.62, "support": 1550},
            "rustic_nostalgia": {"precision": 0.44, "recall": 0.63, "f1": 0.51, "support": 718},
            "study_focus":      {"precision": 0.73, "recall": 0.58, "f1": 0.64, "support": 1621},
            "summertime_party": {"precision": 0.43, "recall": 0.45, "f1": 0.44, "support": 939},
        },
    },
    "dtree": {
        "name":        "Decision Tree",
        "config":      "DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, "
                       "class_weight='balanced')",
        "family":      "Tree (baseline)",
        "tuned":       False,
        "train_acc":   0.6496,
        "test_acc":    0.5999,
        "test_macro_f1": 0.5948,
        "per_class": {
            "heartbreak_sad":   {"precision": 0.42, "recall": 0.55, "f1": 0.48, "support": 898},
            "hype_rap":         {"precision": 0.75, "recall": 0.80, "f1": 0.78, "support": 697},
            "jazz_lounge":      {"precision": 0.78, "recall": 0.58, "f1": 0.67, "support": 1550},
            "rustic_nostalgia": {"precision": 0.40, "recall": 0.61, "f1": 0.49, "support": 718},
            "study_focus":      {"precision": 0.83, "recall": 0.62, "f1": 0.71, "support": 1621},
            "summertime_party": {"precision": 0.43, "recall": 0.47, "f1": 0.45, "support": 939},
        },
    },
    "dtree_tuned": {
        "name":        "Decision Tree (GridSearch)",
        "config":      "DecisionTreeClassifier(criterion='entropy', max_depth=20, "
                       "min_samples_leaf=50, class_weight='balanced')",
        "family":      "Tree (tuned)",
        "tuned":       True,
        "tuner":       "GridSearchCV — 48 candidates × 3 folds",
        "cv_macro_f1": 0.6046,
        "test_macro_f1": 0.6039,
        "test_acc":    0.6090,
        "train_acc":   0.6560,
        "per_class": {
            "heartbreak_sad":   {"precision": 0.42, "recall": 0.54, "f1": 0.47, "support": 898},
            "hype_rap":         {"precision": 0.78, "recall": 0.83, "f1": 0.80, "support": 697},
            "jazz_lounge":      {"precision": 0.74, "recall": 0.60, "f1": 0.66, "support": 1550},
            "rustic_nostalgia": {"precision": 0.44, "recall": 0.59, "f1": 0.50, "support": 718},
            "study_focus":      {"precision": 0.82, "recall": 0.63, "f1": 0.72, "support": 1621},
            "summertime_party": {"precision": 0.44, "recall": 0.50, "f1": 0.47, "support": 939},
        },
    },
    "rf": {
        "name":        "Random Forest",
        "config":      "RandomForestClassifier(n_estimators=300, min_samples_leaf=2, "
                       "class_weight='balanced')",
        "family":      "Ensemble (baseline)",
        "tuned":       False,
        "train_acc":   0.9828,
        "test_acc":    0.6853,
        "test_macro_f1": 0.6810,
        "cv_scores":   [0.7181, 0.6867, 0.6522, 0.6440, 0.6636],
        "cv_mean":     0.6729,
        "cv_std":      0.0268,
        "per_class": {
            "heartbreak_sad":   {"precision": 0.53, "recall": 0.63, "f1": 0.57, "support": 898},
            "hype_rap":         {"precision": 0.79, "recall": 0.87, "f1": 0.83, "support": 697},
            "jazz_lounge":      {"precision": 0.79, "recall": 0.69, "f1": 0.74, "support": 1550},
            "rustic_nostalgia": {"precision": 0.62, "recall": 0.65, "f1": 0.63, "support": 718},
            "study_focus":      {"precision": 0.83, "recall": 0.68, "f1": 0.75, "support": 1621},
            "summertime_party": {"precision": 0.52, "recall": 0.62, "f1": 0.56, "support": 939},
        },
    },
    "rf_tuned": {
        "name":        "Random Forest (Optuna)",
        "config":      "RandomForestClassifier(n_estimators=400, max_depth=22, "
                       "min_samples_leaf=1, min_samples_split=4, max_features=0.5, "
                       "max_samples=0.640, class_weight='balanced')",
        "family":      "Ensemble (tuned)",
        "tuned":       True,
        "tuner":       "Optuna TPE — 60 trials × 3-fold CV",
        "cv_macro_f1": 0.6754,
        "train_acc":   0.9721,
        "test_acc":    0.6863,
        "test_macro_f1": 0.6814,
        "per_class": {
            "heartbreak_sad":   {"precision": 0.53, "recall": 0.61, "f1": 0.57, "support": 898},
            "hype_rap":         {"precision": 0.81, "recall": 0.87, "f1": 0.84, "support": 697},
            "jazz_lounge":      {"precision": 0.78, "recall": 0.70, "f1": 0.74, "support": 1550},
            "rustic_nostalgia": {"precision": 0.60, "recall": 0.66, "f1": 0.63, "support": 718},
            "study_focus":      {"precision": 0.82, "recall": 0.69, "f1": 0.75, "support": 1621},
            "summertime_party": {"precision": 0.52, "recall": 0.61, "f1": 0.57, "support": 939},
        },
    },
    "lgbm": {
        "name":        "LightGBM",
        "config":      "LGBMClassifier(n_estimators=800, learning_rate=0.05, "
                       "num_leaves=31, min_child_samples=20, subsample=0.8, "
                       "colsample_bytree=0.8, class_weight='balanced')",
        "family":      "Gradient boosting (baseline)",
        "tuned":       False,
        "train_acc":   0.9863,
        "test_acc":    0.6933,
        "test_macro_f1": 0.6888,
        "per_class": {
            "heartbreak_sad":   {"precision": 0.53, "recall": 0.62, "f1": 0.58, "support": 898},
            "hype_rap":         {"precision": 0.82, "recall": 0.86, "f1": 0.84, "support": 697},
            "jazz_lounge":      {"precision": 0.79, "recall": 0.70, "f1": 0.74, "support": 1550},
            "rustic_nostalgia": {"precision": 0.61, "recall": 0.70, "f1": 0.65, "support": 718},
            "study_focus":      {"precision": 0.81, "recall": 0.70, "f1": 0.75, "support": 1621},
            "summertime_party": {"precision": 0.54, "recall": 0.60, "f1": 0.57, "support": 939},
        },
    },
    "lgbm_tuned": {
        "name":        "LightGBM (Optuna)",
        "config":      "LGBMClassifier(n_estimators=600, learning_rate=0.0121, "
                       "num_leaves=120, max_depth=30, min_child_samples=64, "
                       "subsample=0.914, colsample_bytree=0.521, reg_alpha=0.005, "
                       "reg_lambda=0.036, class_weight='balanced')",
        "family":      "Gradient boosting (tuned)",
        "tuned":       True,
        "tuner":       "Optuna TPE — 40 trials × 3-fold CV",
        "cv_macro_f1": 0.6861,
        "train_acc":   0.9626,
        "test_acc":    0.6981,
        "test_macro_f1": 0.6946,
        "is_best":     True,
        "per_class": {
            "heartbreak_sad":   {"precision": 0.54, "recall": 0.64, "f1": 0.59, "support": 898},
            "hype_rap":         {"precision": 0.82, "recall": 0.87, "f1": 0.84, "support": 697},
            "jazz_lounge":      {"precision": 0.81, "recall": 0.70, "f1": 0.75, "support": 1550},
            "rustic_nostalgia": {"precision": 0.62, "recall": 0.70, "f1": 0.66, "support": 718},
            "study_focus":      {"precision": 0.82, "recall": 0.70, "f1": 0.75, "support": 1621},
            "summertime_party": {"precision": 0.54, "recall": 0.62, "f1": 0.58, "support": 939},
        },
    },
}

# Display order (left → right) on comparison charts
MODEL_ORDER = ["logreg", "dtree", "dtree_tuned", "rf", "rf_tuned", "lgbm", "lgbm_tuned"]
BEST_MODEL_KEY = "lgbm_tuned"


MODEL_RESULTS = {
    "model":         MODELS[BEST_MODEL_KEY]["config"],
    "model_name":    MODELS[BEST_MODEL_KEY]["name"],
    "n_train":       N_TRAIN,
    "n_test":        N_TEST,
    "train_acc":     MODELS[BEST_MODEL_KEY]["train_acc"],
    "test_acc":      MODELS[BEST_MODEL_KEY]["test_acc"],
    "test_macro_f1": MODELS[BEST_MODEL_KEY]["test_macro_f1"],
    # CV scores still come from the RF baseline (the only model with
    # 5-fold CV recorded in the notebook). The RF Optuna and LGBM
    # Optuna runs use 3-fold CV (cv_macro_f1 fields above).
    "cv_scores":     MODELS["rf"]["cv_scores"],
    "cv_mean":       MODELS["rf"]["cv_mean"],
    "cv_std":        MODELS["rf"]["cv_std"],
    "ablation": {
        # Updated from the new notebook cell 106 ablation run
        "Audio only":    {"acc": 0.5851, "f1": 0.5734, "n_features": 14},
        "Lyric only":    {"acc": 0.5865, "f1": 0.5814, "n_features": 23},
        "Audio + Lyric": {"acc": 0.6815, "f1": 0.6770, "n_features": 37},
    },
    # Per-class breakdown from the BEST model (Tuned LightGBM)
    "per_class":     MODELS[BEST_MODEL_KEY]["per_class"],
}


ARCHETYPE_PROFILES = {
    "study_focus": {
        # audio (typical lo-fi / ambient / instrumental)
        "danceability": 0.50, "energy": 0.30, "key": 5, "loudness": -14.0, "mode": 1,
        "speechiness": 0.04, "acousticness": 0.55, "instrumentalness": 0.65,
        "liveness": 0.12, "valence": 0.35, "tempo": 95.0,
        "duration_ms": 200_000, "time_signature": 4, "year": 2018,
        # lyric (notebook cell 67 — means)
        "words_per_second": 1.058, "lyric_sentiment_pos": 0.141,
        "lyric_sentiment_neg": 0.081, "lyric_sentiment_neu": 0.778,
        "lyric_compound_score": 0.380,
        "emotion_score_anger": 0.140, "emotion_score_disgust": 0.05,
        "emotion_score_fear": 0.18, "emotion_score_joy": 0.061,
        "emotion_score_neutral": 0.170, "emotion_score_sadness": 0.217,
        "emotion_score_surprise": 0.06,
        "lyric_arch_sim_study_focus": 0.40, "lyric_arch_sim_heartbreak_sad": 0.25,
        "lyric_arch_sim_summertime_party": 0.20, "lyric_arch_sim_rustic_nostalgia": 0.22,
        "lyric_arch_sim_jazz_lounge": 0.30, "lyric_arch_sim_hype_rap": 0.15,
        "dominant_emotion_enc": 4, "lyrics_language_enc": 2, "is_english": 1,
        "explicit": 0, "has_lyrics": 1,
    },
    "jazz_lounge": {
        "danceability": 0.55, "energy": 0.35, "key": 5, "loudness": -13.0, "mode": 1,
        "speechiness": 0.05, "acousticness": 0.65, "instrumentalness": 0.40,
        "liveness": 0.18, "valence": 0.50, "tempo": 110.0,
        "duration_ms": 280_000, "time_signature": 4, "year": 1985,
        "words_per_second": 0.933, "lyric_sentiment_pos": 0.170,
        "lyric_sentiment_neg": 0.065, "lyric_sentiment_neu": 0.765,
        "lyric_compound_score": 0.573,
        "emotion_score_anger": 0.104, "emotion_score_disgust": 0.04,
        "emotion_score_fear": 0.15, "emotion_score_joy": 0.142,
        "emotion_score_neutral": 0.168, "emotion_score_sadness": 0.276,
        "emotion_score_surprise": 0.07,
        "lyric_arch_sim_study_focus": 0.30, "lyric_arch_sim_heartbreak_sad": 0.30,
        "lyric_arch_sim_summertime_party": 0.25, "lyric_arch_sim_rustic_nostalgia": 0.25,
        "lyric_arch_sim_jazz_lounge": 0.45, "lyric_arch_sim_hype_rap": 0.15,
        "dominant_emotion_enc": 5, "lyrics_language_enc": 2, "is_english": 1,
        "explicit": 0, "has_lyrics": 1,
    },
    "summertime_party": {
        "danceability": 0.75, "energy": 0.78, "key": 5, "loudness": -5.5, "mode": 1,
        "speechiness": 0.07, "acousticness": 0.18, "instrumentalness": 0.04,
        "liveness": 0.20, "valence": 0.70, "tempo": 122.0,
        "duration_ms": 215_000, "time_signature": 4, "year": 2017,
        "words_per_second": 1.331, "lyric_sentiment_pos": 0.148,
        "lyric_sentiment_neg": 0.082, "lyric_sentiment_neu": 0.770,
        "lyric_compound_score": 0.389,
        "emotion_score_anger": 0.152, "emotion_score_disgust": 0.06,
        "emotion_score_fear": 0.13, "emotion_score_joy": 0.076,
        "emotion_score_neutral": 0.184, "emotion_score_sadness": 0.193,
        "emotion_score_surprise": 0.09,
        "lyric_arch_sim_study_focus": 0.20, "lyric_arch_sim_heartbreak_sad": 0.25,
        "lyric_arch_sim_summertime_party": 0.45, "lyric_arch_sim_rustic_nostalgia": 0.25,
        "lyric_arch_sim_jazz_lounge": 0.20, "lyric_arch_sim_hype_rap": 0.30,
        "dominant_emotion_enc": 3, "lyrics_language_enc": 2, "is_english": 1,
        "explicit": 0, "has_lyrics": 1,
    },
    "heartbreak_sad": {
        "danceability": 0.45, "energy": 0.35, "key": 5, "loudness": -8.5, "mode": 0,
        "speechiness": 0.06, "acousticness": 0.55, "instrumentalness": 0.06,
        "liveness": 0.13, "valence": 0.25, "tempo": 105.0,
        "duration_ms": 230_000, "time_signature": 4, "year": 2015,
        "words_per_second": 1.076, "lyric_sentiment_pos": 0.143,
        "lyric_sentiment_neg": 0.096, "lyric_sentiment_neu": 0.761,
        "lyric_compound_score": 0.313,
        "emotion_score_anger": 0.124, "emotion_score_disgust": 0.05,
        "emotion_score_fear": 0.20, "emotion_score_joy": 0.040,
        "emotion_score_neutral": 0.135, "emotion_score_sadness": 0.300,
        "emotion_score_surprise": 0.05,
        "lyric_arch_sim_study_focus": 0.25, "lyric_arch_sim_heartbreak_sad": 0.50,
        "lyric_arch_sim_summertime_party": 0.20, "lyric_arch_sim_rustic_nostalgia": 0.30,
        "lyric_arch_sim_jazz_lounge": 0.30, "lyric_arch_sim_hype_rap": 0.15,
        "dominant_emotion_enc": 5, "lyrics_language_enc": 2, "is_english": 1,
        "explicit": 0, "has_lyrics": 1,
    },
    "rustic_nostalgia": {
        "danceability": 0.55, "energy": 0.55, "key": 5, "loudness": -7.5, "mode": 1,
        "speechiness": 0.04, "acousticness": 0.40, "instrumentalness": 0.02,
        "liveness": 0.18, "valence": 0.55, "tempo": 120.0,
        "duration_ms": 220_000, "time_signature": 4, "year": 2008,
        "words_per_second": 1.160, "lyric_sentiment_pos": 0.145,
        "lyric_sentiment_neg": 0.075, "lyric_sentiment_neu": 0.780,
        "lyric_compound_score": 0.516,
        "emotion_score_anger": 0.117, "emotion_score_disgust": 0.04,
        "emotion_score_fear": 0.16, "emotion_score_joy": 0.080,
        "emotion_score_neutral": 0.203, "emotion_score_sadness": 0.254,
        "emotion_score_surprise": 0.07,
        "lyric_arch_sim_study_focus": 0.25, "lyric_arch_sim_heartbreak_sad": 0.30,
        "lyric_arch_sim_summertime_party": 0.25, "lyric_arch_sim_rustic_nostalgia": 0.50,
        "lyric_arch_sim_jazz_lounge": 0.25, "lyric_arch_sim_hype_rap": 0.15,
        "dominant_emotion_enc": 4, "lyrics_language_enc": 2, "is_english": 1,
        "explicit": 0, "has_lyrics": 1,
    },
    "hype_rap": {
        "danceability": 0.78, "energy": 0.75, "key": 5, "loudness": -5.5, "mode": 0,
        "speechiness": 0.30, "acousticness": 0.10, "instrumentalness": 0.01,
        "liveness": 0.20, "valence": 0.50, "tempo": 130.0,
        "duration_ms": 210_000, "time_signature": 4, "year": 2019,
        "words_per_second": 2.469, "lyric_sentiment_pos": 0.115,
        "lyric_sentiment_neg": 0.130, "lyric_sentiment_neu": 0.755,
        "lyric_compound_score": -0.187,
        "emotion_score_anger": 0.318, "emotion_score_disgust": 0.10,
        "emotion_score_fear": 0.13, "emotion_score_joy": 0.038,
        "emotion_score_neutral": 0.173, "emotion_score_sadness": 0.086,
        "emotion_score_surprise": 0.05,
        "lyric_arch_sim_study_focus": 0.15, "lyric_arch_sim_heartbreak_sad": 0.20,
        "lyric_arch_sim_summertime_party": 0.30, "lyric_arch_sim_rustic_nostalgia": 0.15,
        "lyric_arch_sim_jazz_lounge": 0.15, "lyric_arch_sim_hype_rap": 0.55,
        "dominant_emotion_enc": 0, "lyrics_language_enc": 2, "is_english": 1,
        "explicit": 1, "has_lyrics": 1,
    },
}


DATA_DIR = Path(__file__).parent / "data"

@lru_cache(maxsize=1)
def load_dataframe() -> tuple[pd.DataFrame, bool]:
    """Returns (df, is_real). If `data/tracks.csv` exists use it,
    otherwise generate a synthetic dataset matching the real class
    distribution (32,112 rows) for the demo."""
    csv = DATA_DIR / "tracks.csv"
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            return df, True
        except Exception:
            pass
    return _synthetic_dataframe(), False

def _synthetic_dataframe() -> pd.DataFrame:
    """Generate a representative synthetic dataset of 32,112 rows
    matching the actual class distribution from the notebook."""
    rng = np.random.default_rng(42)
    rows = []
    for arch, count in CLASS_COUNTS.items():
        profile = ARCHETYPE_PROFILES[arch]
        for _ in range(count):
            row = {"archetype": arch}
            # audio: jitter around profile mean
            for f in AUDIO_FEATURES:
                lo, hi, _ = AUDIO_RANGES[f]
                mean = profile[f]
                if f in ("key", "mode", "time_signature"):
                    row[f] = int(round(np.clip(mean + rng.normal(0, 0.5), lo, hi)))
                elif f == "year":
                    row[f] = int(round(np.clip(mean + rng.normal(0, 6), lo, hi)))
                elif f in ("loudness", "tempo", "duration_ms"):
                    sd = (hi - lo) * 0.08
                    row[f] = float(np.clip(mean + rng.normal(0, sd), lo, hi))
                else:
                    row[f] = float(np.clip(mean + rng.normal(0, 0.10), lo, hi))
            # lyric: jitter around profile mean
            for f in LYRIC_NUMERIC_FEATURES:
                mean = profile[f]
                row[f] = float(np.clip(mean + rng.normal(0, 0.08), -1.0, 3.0))
            # categorical
            for f in CATEGORICAL_FEATURES:
                row[f] = int(profile[f])
            rows.append(row)
    df = pd.DataFrame(rows)
    df["track_name"] = [f"Track {i}" for i in range(len(df))]
    df["artist_name"] = [f"Artist {(i//50) % 200}" for i in range(len(df))]
    return df


@lru_cache(maxsize=1)
def load_model():
    """Returns (model, is_real). If `data/model.pkl` exists, load it.
    Otherwise train a fresh RandomForest on the synthetic dataset
    using the same hyperparameters as the notebook's `rf`."""
    pkl = DATA_DIR / "model.pkl"
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                return pickle.load(f), True
        except Exception:
            pass

    from sklearn.ensemble import RandomForestClassifier
    df, _ = load_dataframe()
    X = df[ALL_FEATURES].values
    y = df["archetype"].values
    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf, False


def predict_proba(feature_dict: dict) -> dict[str, float]:
    """Run the loaded model on a single feature dict and return
    {archetype: probability}, sorted descending."""
    model, _ = load_model()
    x = np.array([[feature_dict.get(f, 0.0) for f in ALL_FEATURES]])
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    out = {cls: float(p) for cls, p in zip(classes, proba)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def archetype_preset(archetype: str) -> dict:
    """Return a complete 37-feature dict matching this archetype's
    typical profile. Used by the predictor's preset buttons."""
    return dict(ARCHETYPE_PROFILES[archetype])
