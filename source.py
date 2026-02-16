"""
source.py
---------
Core functions for the Streamlit app: "Financial Text Sentiment Analysis: From Bag-of-Words to FinBERT".

Design goals
- Notebook logic is wrapped into functions so Streamlit can import and cache/reuse computations.
- Includes safe fallbacks (dummy data) so the app still runs even when external downloads are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Stats
from scipy.stats import spearmanr

# Transformers (FinBERT)
try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


# -------------------------------------------------------------------
# NLTK setup (safe: download if missing)
# -------------------------------------------------------------------
def _safe_nltk_download() -> None:
    """
    Ensure required NLTK assets exist.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


_safe_nltk_download()


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_initial_data(
    phrasebank_dataset: str = "descartes100/enhanced-financial-phrasebank",
    tenk_path: str = "apple_10k_risk_factors.txt",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load:
    1) Financial PhraseBank (labeled sentiment)
    2) SEC 10-K risk factor paragraphs (unlabeled, real-world text)

    Returns:
        df_fpb: columns ['text','label','sentiment']
        df_10k: columns ['text']
        risk_factor_paragraphs: list[str]
    """
    # --- Financial PhraseBank ---
    df_fpb: pd.DataFrame
    try:
        # Load from CSV file
        csv_path = Path("financial_phrasebank.csv")
        df_fpb = pd.read_csv(csv_path)

        # IMPORTANT: Limit to first 2423 rows to match cached FinBERT predictions
        # The cached predictions were generated from the original 2423-row dataset
        if len(df_fpb) > 2423:
            print(
                f"Note: Loaded {len(df_fpb)} rows from CSV, but limiting to first 2423 to match cached FinBERT predictions.")
            df_fpb = df_fpb.head(2423).copy()

        # Normalize expected columns
        if "sentence" in df_fpb.columns and "text" not in df_fpb.columns:
            df_fpb = df_fpb.rename(columns={"sentence": "text"})
        if "label" not in df_fpb.columns:
            # Some datasets store labels differently; attempt to recover.
            # If still missing, fail to dummy data.
            raise ValueError(
                "Expected 'label' column not found in PhraseBank dataset.")

        # If label is numeric, map to sentiment strings
        if df_fpb["label"].dtype in ['int64', 'int32', 'float64', 'float32']:
            sentiment_to_label = {0: "negative", 1: "neutral", 2: "positive"}
            df_fpb["sentiment"] = df_fpb["label"].map(sentiment_to_label)
        elif "sentiment" not in df_fpb.columns:
            # If label is already a string, use it as sentiment
            df_fpb["sentiment"] = df_fpb["label"]

        df_fpb = df_fpb[["text", "label", "sentiment"]].copy()
        df_fpb["text"] = df_fpb["text"].astype(str)

    except Exception as e:
        # Fallback dummy data (keeps the app functional offline)
        print("Warning: Could not load Financial PhraseBank dataset. Using dummy data instead.")
        dummy_data = {
            "text": [
                "The company reported a record profit increase of 20% in the last quarter.",
                "Revenue was largely flat, showing no significant change year-over-year.",
                "Despite cost-cutting measures, the firm announced a substantial loss.",
                "Expectations for future growth remain positive due to new market opportunities.",
                "The market reaction was neutral following the CEO's vague statements.",
                "Increased competition led to a sharp decline in market share.",
                "Strong earnings beat analyst estimates significantly.",
                "The outlook remains unchanged from previous guidance.",
                "Losses deepened as restructuring costs mounted.",
                "Management expressed confidence in the strategic plan.",
                "Trading volume showed no notable patterns.",
                "The company faces mounting pressure from creditors.",
                "New product launch exceeded sales targets.",
                "Market share remained stable quarter over quarter.",
                "Revenue declined sharply due to weak demand.",
                "Investors welcomed the dividend increase announcement.",
                "The financial results were in line with expectations.",
                "The company issued a profit warning for next quarter.",
                "Strong operational performance drove margin expansion.",
                "Business conditions remained steady with no major changes.",
                "The firm reported disappointing quarterly results.",
                "Growth accelerated in key international markets.",
                "Performance metrics showed neither improvement nor decline.",
                "Cost overruns impacted profitability negatively.",
                "The company announced a major strategic acquisition.",
                "Sales figures came in as anticipated by analysts.",
                "Operating losses widened amid challenging conditions.",
                "Strong customer demand boosted revenue growth.",
                "The business outlook remained cautious but stable.",
                "The company struggled with supply chain disruptions.",
            ],
            "label": [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0],
            "sentiment": ["positive", "neutral", "negative", "positive", "neutral", "negative",
                          "positive", "neutral", "negative", "positive", "neutral", "negative",
                          "positive", "neutral", "negative", "positive", "neutral", "negative",
                          "positive", "neutral", "negative", "positive", "neutral", "negative",
                          "positive", "neutral", "negative", "positive", "neutral", "negative"],
        }
        df_fpb = pd.DataFrame(dummy_data)

    # --- 10-K risk factors ---
    csv_10k_path = Path("sec_10k.csv")
    try:
        df_10k = pd.read_csv(csv_10k_path)
        # Extract just the text column if the CSV has predictions
        if "text" not in df_10k.columns:
            raise ValueError("Expected 'text' column not found in sec_10k.csv")
        risk_factor_paragraphs = df_10k["text"].tolist()
    except Exception:
        # Fallback to text file or dummy data
        tenk_file = Path(tenk_path)
        if tenk_file.exists():
            risk_text = tenk_file.read_text(encoding="utf-8")
        else:
            risk_text = """
Our business depends on the continued service of certain key employees. If we are unable to attract or retain qualified personnel, our business could be harmed.
The global economy and capital markets are subject to periods of disruption and volatility. Adverse changes in economic conditions could negatively impact our financial condition and results of operations.
We are subject to intense competition, which could adversely affect our business and operating results.
Changes in effective tax rates, tax laws, and taxation of our international operations could harm our business.
Our products and services may experience defects or performance problems, which could harm our reputation and results of operations.
Compliance with new and existing laws and regulations could increase our costs and adversely affect our business.
""".strip()
            tenk_file.write_text(risk_text, encoding="utf-8")

        risk_factor_paragraphs = [
            p.strip() for p in risk_text.split("\n") if len(p.strip()) > 50]
        df_10k = pd.DataFrame({"text": risk_factor_paragraphs})

    return df_fpb, df_10k, risk_factor_paragraphs


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------
_FINANCE_KEEP_STOPWORDS = {
    # Negation and comparison words are crucial in finance
    "not",
    "no",
    "nor",
    "never",
    "none",
    "without",
    "below",
    "above",
    "under",
    "over",
    "down",
    "up",
    "less",
    "more",
    "against",
}


def preprocess_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Domain-aware preprocessing for financial text.

    Steps:
    - Lowercase
    - Preserve key finance symbols (% and $), strip other punctuation
    - Tokenize
    - Optionally remove stopwords, but preserve negation/comparison words

    Returns:
        tokens: list[str]
    """
    if text is None:
        return []

    text = str(text).lower()

    # Keep % and $ because they carry meaning in finance
    text = re.sub(r"[^a-z0-9\s\%\$]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    if not remove_stopwords:
        return tokens

    sw = set(stopwords.words("english"))
    sw = sw - _FINANCE_KEEP_STOPWORDS

    tokens = [t for t in tokens if t not in sw]
    return tokens


# -------------------------------------------------------------------
# Loughran–McDonald (simplified demo lists) + scoring
# -------------------------------------------------------------------
lm_positive = {
    "achieve", "attain", "benefit", "better", "boost", "creative", "efficiency",
    "enhance", "excellent", "exceed", "favorable", "gain", "great", "improve",
    "innovation", "opportunity", "optimistic", "outperform", "positive", "profit",
    "progress", "record", "rebound", "recovery", "strength", "strong", "succeed",
    "surpass", "upgrade", "upturn", "advantage", "growth", "solid", "stronger", "well"
}

lm_negative = {
    "adverse", "against", "breakdown", "burden", "claim", "closure", "concern",
    "decline", "default", "deficit", "deteriorate", "disappoint", "downturn",
    "failure", "fall", "fraud", "impair", "investigation", "layoff", "litigation",
    "loss", "negative", "penalty", "plunge", "problem", "recall", "restructuring",
    "risk", "shortfall", "slowdown", "sued", "terminate", "threat", "unable",
    "unfavorable", "violation", "weak", "worse", "writedown", "writeoff", "challenging",
    "uncertainty", "volatility", "lower", "decrease", "recession"
}


def lm_sentiment_score(
    text_tokens: Sequence[str],
    pos_words: set[str] = lm_positive,
    neg_words: set[str] = lm_negative,
) -> Tuple[float, int, int, int, str]:
    """
    Returns:
        (score, n_pos, n_neg, n_total, predicted_sentiment)
    """
    n_pos = sum(1 for t in text_tokens if t in pos_words)
    n_neg = sum(1 for t in text_tokens if t in neg_words)
    n_total = max(len(text_tokens), 1)
    score = (n_pos - n_neg) / n_total

    if score > 0.02:
        pred = "positive"
    elif score < -0.02:
        pred = "negative"
    else:
        pred = "neutral"
    return score, n_pos, n_neg, n_total, pred


# -------------------------------------------------------------------
# TF-IDF + Logistic Regression
# -------------------------------------------------------------------
def train_tfidf_logreg_model(
    df_fpb_processed: pd.DataFrame,
    text_col_raw: str = "text",
    text_col_processed: str = "processed_text",
    label_col: str = "sentiment",
    test_size: float = 0.5,
    random_state: int = 42,
):
    """
    Train TF-IDF + Logistic Regression.

    Returns:
        tfidf_vectorizer, lr_model,
        X_train_raw, X_test_raw,
        y_train, y_test,
        feature_names, sentiment_classes
    """
    if df_fpb_processed is None or df_fpb_processed.empty:
        raise ValueError(
            "PhraseBank dataframe is empty; cannot train TF-IDF model.")

    X_raw = df_fpb_processed[text_col_raw].astype(str)
    X_proc = df_fpb_processed[text_col_processed].astype(str)
    y = df_fpb_processed[label_col].astype(str)

    # Ensure minimum test size for stratified split
    n_classes = y.nunique()
    # At least n_classes or 10% of data
    min_test_samples = max(n_classes, int(len(y) * 0.1))
    actual_test_size = max(test_size, min_test_samples / len(y))

    # Use processed text for vectorization but keep raw split for FinBERT comparison later
    X_train_proc, X_test_proc, y_train, y_test, X_train_raw, X_test_raw = train_test_split(
        X_proc,
        y,
        X_raw,
        test_size=actual_test_size,
        random_state=random_state,
        stratify=y if n_classes > 1 and len(y) >= 2 * n_classes else None,
    )

    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_proc)
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    lr_model.fit(X_train_tfidf, y_train)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    sentiment_classes = lr_model.classes_

    return (
        tfidf_vectorizer,
        lr_model,
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        feature_names,
        sentiment_classes,
    )


def predict_tfidf_logreg(
    df: pd.DataFrame,
    processed_text_col: str,
    tfidf_vectorizer: TfidfVectorizer,
    lr_model: LogisticRegression,
    out_col: str = "tfidf_pred",
) -> pd.DataFrame:
    """
    Add TF-IDF + LogReg predictions to a dataframe.
    """
    out = df.copy()
    X_tfidf = tfidf_vectorizer.transform(out[processed_text_col].astype(str))
    out[out_col] = lr_model.predict(X_tfidf)
    return out


# -------------------------------------------------------------------
# FinBERT - Predictions loaded from JSON files
# -------------------------------------------------------------------
# Note: FinBERT model loading and prediction functions have been removed
# for performance. Predictions are loaded directly from pre-computed JSON files.


# -------------------------------------------------------------------
# Evaluation (comparison)
# -------------------------------------------------------------------
def evaluate_all_models(
    comparison_df: pd.DataFrame,
    y_true: Sequence[str],
    models_to_evaluate_preds: Dict[str, Sequence[str]],
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Compute Accuracy, Macro-F1, and per-class F1s, plus confusion matrices figure.

    Returns:
        performance_table: pd.DataFrame indexed by model name
        fig: matplotlib Figure
    """
    results = {}
    for name, preds in models_to_evaluate_preds.items():
        acc = accuracy_score(y_true, preds)
        macro = f1_score(y_true, preds, average="macro")
        report = classification_report(
            y_true, preds, output_dict=True, zero_division=0)

        results[name] = {
            "Accuracy": acc,
            "Macro-F1": macro,
            "F1 (Positive)": report.get("positive", {}).get("f1-score", 0.0),
            "F1 (Negative)": report.get("negative", {}).get("f1-score", 0.0),
            "F1 (Neutral)": report.get("neutral", {}).get("f1-score", 0.0),
        }

    performance_table = pd.DataFrame(results).T

    labels_order = ["negative", "neutral", "positive"]
    fig, axes = plt.subplots(1, len(models_to_evaluate_preds), figsize=(18, 5))
    if len(models_to_evaluate_preds) == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices: Three Sentiment Approaches", fontsize=14)

    for ax, (name, preds) in zip(axes, models_to_evaluate_preds.items()):
        cm = confusion_matrix(y_true, preds, labels=labels_order)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            ax=ax,
            xticklabels=labels_order,
            yticklabels=labels_order,
            cbar=False,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return performance_table, fig


# -------------------------------------------------------------------
# Sentiment–return correlation (conceptual, simulated)
# -------------------------------------------------------------------
def run_sentiment_return_correlation_analysis(
    seed: int = 42,
    num_days: int = 252,
    tickers: Optional[List[str]] = None,
):
    """
    Simulate sentiment + next-day returns, then:
    - Spearman correlation between avg sentiment and next-day return
    - Quintile spread analysis (Q5 - Q1), annualized, in bps

    Returns:
        corr, p_val,
        annualized_quintile_returns_bps (Series),
        long_short_spread_bps (float),
        fig (matplotlib Figure)
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    np.random.seed(seed)
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=num_days))
    rows = []

    for date in dates:
        for t in tickers:
            finbert_score = np.random.uniform(-0.8, 0.8)
            base_ret = np.random.normal(0, 0.005)
            next_day_return = base_ret + (finbert_score * 0.001)
            rows.append(
                {
                    "date": date,
                    "ticker": t,
                    "headline": f"News for {t} on {date.strftime('%Y-%m-%d')}",
                    "finbert_score": finbert_score,
                    "next_day_return": next_day_return,
                }
            )

    news_df = pd.DataFrame(rows)

    daily_sent_agg = (
        news_df.groupby(["date", "ticker"])
        .agg(avg_sentiment=("finbert_score", "mean"), next_day_return=("next_day_return", "first"))
        .reset_index()
    )

    corr, p_val = spearmanr(
        daily_sent_agg["avg_sentiment"], daily_sent_agg["next_day_return"])

    daily_sent_agg["sent_quintile"] = pd.qcut(
        daily_sent_agg["avg_sentiment"], q=5, labels=[1, 2, 3, 4, 5]
    )

    quintile_returns = daily_sent_agg.groupby(
        "sent_quintile")["next_day_return"].mean()
    annualized_quintile_returns_bps = quintile_returns * 252 * 10000
    long_short_spread_bps = float(
        annualized_quintile_returns_bps.loc[5] - annualized_quintile_returns_bps.loc[1])

    fig = plt.figure(figsize=(10, 6))
    annualized_quintile_returns_bps.plot(kind="bar")
    plt.title(
        "Annualized Returns by Sentiment Quintile (Simulated Data)", fontsize=14)
    plt.xlabel(
        "Sentiment Quintile (1 = Most Negative, 5 = Most Positive)", fontsize=12)
    plt.ylabel("Annualized Return (bps)", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.tight_layout()

    return corr, p_val, annualized_quintile_returns_bps, long_short_spread_bps, fig
