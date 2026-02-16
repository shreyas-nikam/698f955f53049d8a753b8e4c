import os
import json
import hashlib
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

# Import all custom functions and variables from source.py
from source import (
    load_initial_data,
    preprocess_text,
    lm_positive,
    lm_negative,
    lm_sentiment_score,
    train_tfidf_logreg_model,
    predict_tfidf_logreg,
    evaluate_all_models,
    run_sentiment_return_correlation_analysis,
)

# -------------------------------------------------------------------
# Optional FinBERT support (fallback if not implemented in source.py)
# -------------------------------------------------------------------
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None


def _cache_dir() -> Path:
    p = Path(".cache_fin_sentiment")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def _load_finbert_pipeline():
    """
    Loads FinBERT pipeline if transformers is available.
    Cached as a Streamlit resource to avoid repeated loads.
    """
    if hf_pipeline is None:
        return None
    return hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1,  # CPU default
    )


def finbert_predict_batch(texts, batch_size=32):
    """
    Predict FinBERT sentiment for a list of texts with disk caching.

    Returns: list[str] with labels in {"positive","negative","neutral"}.
    """
    texts = list(texts)
    cache_key = _hash_texts(texts)
    cache_path = _cache_dir() / f"finbert_{cache_key}.json"

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    pipe = _load_finbert_pipeline()
    if pipe is None:
        raise RuntimeError(
            "FinBERT requires the 'transformers' library. "
            "Install it, or disable FinBERT features."
        )

    out_labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        preds = pipe(batch, truncation=True)
        for p in preds:
            # HF label mapping for ProsusAI/finbert is typically POSITIVE/NEGATIVE/NEUTRAL
            label = str(p.get("label", "")).lower()
            if "pos" in label:
                out_labels.append("positive")
            elif "neg" in label:
                out_labels.append("negative")
            else:
                out_labels.append("neutral")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(out_labels, f)

    return out_labels


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    layout="wide", page_title="Financial Text Sentiment Analysis")


# -------------------------------------------------------------------
# Session state initialization
# -------------------------------------------------------------------
def initialize_session_state():
    # Navigation
    if "page" not in st.session_state:
        st.session_state.page = "Introduction"

    # Data
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df_financial_phrasebank" not in st.session_state:
        st.session_state.df_financial_phrasebank = None
    if "df_10k" not in st.session_state:
        st.session_state.df_10k = None
    if "risk_factor_paragraphs" not in st.session_state:
        st.session_state.risk_factor_paragraphs = None

    # Preprocessing
    if "preprocessing_done" not in st.session_state:
        st.session_state.preprocessing_done = False
    if "remove_stopwords_checkbox" not in st.session_state:
        st.session_state.remove_stopwords_checkbox = True
    if "df_financial_phrasebank_processed" not in st.session_state:
        st.session_state.df_financial_phrasebank_processed = None
    if "df_10k_processed" not in st.session_state:
        st.session_state.df_10k_processed = None

    # LM
    if "lm_model_run" not in st.session_state:
        st.session_state.lm_model_run = False

    # TF-IDF + LogReg
    if "tfidf_logreg_trained" not in st.session_state:
        st.session_state.tfidf_logreg_trained = False
    if "tfidf_vectorizer" not in st.session_state:
        st.session_state.tfidf_vectorizer = None
    if "lr_model" not in st.session_state:
        st.session_state.lr_model = None
    if "X_train_fpb" not in st.session_state:
        st.session_state.X_train_fpb = None
    if "X_test_fpb" not in st.session_state:
        st.session_state.X_test_fpb = None
    if "y_train_fpb" not in st.session_state:
        st.session_state.y_train_fpb = None
    if "y_test_fpb" not in st.session_state:
        st.session_state.y_test_fpb = None
    if "lr_feature_names" not in st.session_state:
        st.session_state.lr_feature_names = None
    if "lr_sentiment_classes" not in st.session_state:
        st.session_state.lr_sentiment_classes = None
    if "lr_top_words" not in st.session_state:
        st.session_state.lr_top_words = {
            "positive": [], "negative": [], "neutral": []}

    # FinBERT
    if "finbert_loaded" not in st.session_state:
        st.session_state.finbert_loaded = False
    if "finbert_batch_size" not in st.session_state:
        st.session_state.finbert_batch_size = 32

    # Comparison
    if "models_evaluated" not in st.session_state:
        st.session_state.models_evaluated = False
    if "performance_table" not in st.session_state:
        st.session_state.performance_table = None
    if "confusion_matrices_fig" not in st.session_state:
        st.session_state.confusion_matrices_fig = None
    if "comparison_df" not in st.session_state:
        st.session_state.comparison_df = None

    # Correlation
    if "correlation_done" not in st.session_state:
        st.session_state.correlation_done = False
    if "corr" not in st.session_state:
        st.session_state.corr = None
    if "p_val" not in st.session_state:
        st.session_state.p_val = None
    if "annualized_quintile_returns_bps" not in st.session_state:
        st.session_state.annualized_quintile_returns_bps = None
    if "long_short_spread_bps" not in st.session_state:
        st.session_state.long_short_spread_bps = None
    if "quintile_plot_fig" not in st.session_state:
        st.session_state.quintile_plot_fig = None

    # Custom
    if "custom_text_input" not in st.session_state:
        st.session_state.custom_text_input = (
            "The company reported a strong increase in revenue but warned about rising costs."
        )
    if "custom_text_results" not in st.session_state:
        st.session_state.custom_text_results = None


initialize_session_state()


# -------------------------------------------------------------------
# Small UI helpers (pedagogical scaffolding)
# -------------------------------------------------------------------
def learning_header(title: str, learn: str, why: str):
    st.title(title)
    with st.sidebar.expander("Learning objectives for this page", expanded=True):
        st.markdown(
            f"**What you should learn on this page:** {learn}\n\n**Why this matters in practice:** {why}")


def checkpoint(question: str, options: list[str], key: str, help_text: str = ""):
    with st.expander("Checkpoint (1 minute)", expanded=False):
        st.markdown(f"**Question:** {question}")
        st.radio("Your answer", options, key=key, help=help_text)


def assumptions_box(lines: list[str], title: str = "Assumptions & scope (read before interpreting numbers)"):
    with st.expander(title, expanded=True):
        for ln in lines:
            st.markdown(f"- {ln}")


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.session_state.page = st.sidebar.selectbox(
    "Navigate",
    [
        "Introduction",
        "1. Data Acquisition & Review",
        "2. Text Preprocessing",
        "3. LM Lexicon Model",
        "4. TF-IDF + LogReg Model",
        "5. FinBERT Transformer Model",
        "6. Model Comparison",
        "7. Sentiment-Return Correlation",
        "8. Custom Text Analysis",
    ],
)
st.title("QuLab: Lab 8 - Sentiment from News", )
st.divider()
st.sidebar.markdown("---")


# -------------------------------------------------------------------
# Page rendering
# -------------------------------------------------------------------
if st.session_state.page == "Introduction":
    learning_header(
        "Turning Financial Text Into a Testable Signal",
        "How sentiment becomes a measurable signal—and why different approaches trade off interpretability, performance, and operating cost.",
        "You can’t scale human reading of filings/news; a systematic signal must be testable (metrics) and governable (explainable failure modes).",
    )

    st.markdown(
        """
As a Senior Equity Analyst at AlphaQuant Investments, Sarah constantly seeks innovative ways to generate alpha and manage risk. She knows that approximately 80% of financial information exists as unstructured text – earnings call transcripts, 10-K filings, news articles, and analyst reports. Manually sifting through this deluge is not only time-consuming but also prone to human bias, making it impossible to scale.
"""
    )

    st.warning(
        "Guardrail: if you cite a statistic (e.g., “80%”), the app should either provide a source or clearly label it as a narrative assumption."
    )

    st.markdown(
        """
This application documents Sarah's journey to systematically extract investment signals from financial text using sentiment analysis. She will explore three progressively sophisticated natural language processing (NLP) approaches:

1. **Loughran-McDonald Financial Dictionary:** A rules-based, domain-specific lexicon.
2. **TF-IDF with Logistic Regression:** A traditional machine learning approach that learns patterns from labeled data.
3. **FinBERT Transformer:** A state-of-the-art deep learning model pre-trained on financial text for contextual understanding.
"""
    )

    st.markdown(
        """
Sarah's goal is not just to understand these models but to apply them in a real-world workflow, evaluating their trade-offs in accuracy, interpretability, and computational cost. Ultimately, she aims to demonstrate how text sentiment can inform investment decisions, moving AlphaQuant beyond qualitative assessments to systematic, data-driven insights.
"""
    )

    st.markdown("**Micro-example:** Two firms both report +5% revenue growth. One says “margin pressure persists,” the other says “margin expansion continues.” Your signal should distinguish *growth quality* vs *growth risk*.")

    with st.expander("Checkpoint (1 minute)", expanded=False):
        st.markdown(
            "**Question:** Which output would most increase your trust before using sentiment in a workflow?")
        st.radio("Your answer", ["Only a single example prediction", "A confusion matrix by class", "Only headline accuracy"], key="cp_intro",
                 help="Skeptical users usually want error anatomy (false negatives vs false positives), not a single summary number.")
        if st.session_state.get("cp_intro"):
            if st.session_state.cp_intro == "A confusion matrix by class":
                st.success("✓ Correct! A confusion matrix reveals error types (false positives vs false negatives) which is critical for governance. You need to know *how* the model fails, not just *that* it fails.")
            else:
                st.info("Consider: A single accuracy number or example doesn't show you which classes are confused or what the failure modes are. Confusion matrices provide the error anatomy needed for model risk assessment.")

elif st.session_state.page == "1. Data Acquisition & Review":
    learning_header(
        "1. Load Benchmark + Real‑World Text (2 datasets)",
        "Why you need both labeled benchmark data (to measure model quality) and real-world text (to judge decision usefulness).",
        "A model can score well on a benchmark and still fail on 10‑K boilerplate; you want both measurement and realism.",
    )

    st.markdown(
        """
Sarah knows that high-quality data is the bedrock of any robust analysis. For this project, she needs both labeled data to train and evaluate her models and unlabeled, real-world financial text to apply her findings. She'll start by loading the Financial PhraseBank, a benchmark dataset for financial sentiment, and then prepare a set of SEC 10-K risk factor excerpts, which are crucial for understanding a company's potential vulnerabilities.

The Financial PhraseBank provides sentences from financial news with expert sentiment labels, allowing her to quantitatively assess model performance. The 10-K risk factors, on the other hand, represent the kind of raw, complex, and often ambiguous text an analyst encounters daily. Applying models to this unlabeled data will demonstrate their practical utility.
"""
    )

    assumptions_box(
        [
            "PhraseBank unit = 1 sentence with an expert-provided sentiment label.",
            "10‑K unit = 1 risk-factor paragraph (unlabeled; used for qualitative audit and workflow realism).",
            "You should expect different base rates: filings often skew cautious/negative due to legal language.",
        ]
    )

    if st.button("Load Benchmark + 10‑K Text", disabled=st.session_state.data_loaded):
        with st.spinner("Loading data..."):
            try:
                df_fpb, df_10k_raw, risk_paragraphs = load_initial_data()
                st.session_state.df_financial_phrasebank = df_fpb
                st.session_state.df_10k = df_10k_raw
                st.session_state.risk_factor_paragraphs = risk_paragraphs
                st.session_state.data_loaded = True
                st.success("Data loaded successfully.")
            except Exception as e:
                st.error(
                    f"Error loading data: {e}. Ensure required assets are available or generated by source.py."
                )

    if st.session_state.data_loaded:
        st.subheader("Benchmark: Financial PhraseBank (Labeled)")
        st.write(
            f"Dataset size: {len(st.session_state.df_financial_phrasebank)} sentences")
        st.dataframe(st.session_state.df_financial_phrasebank.head())

        st.markdown("**Class balance (why Macro‑F1 matters):**")
        label_counts = st.session_state.df_financial_phrasebank["sentiment"].value_counts(
        )
        st.dataframe(label_counts)

        # Finance-native anchoring: include one ambiguous example
        try:
            pos_ex = st.session_state.df_financial_phrasebank[
                st.session_state.df_financial_phrasebank["sentiment"] == "positive"
            ]["text"].iloc[0]
            neg_ex = st.session_state.df_financial_phrasebank[
                st.session_state.df_financial_phrasebank["sentiment"] == "negative"
            ]["text"].iloc[0]
            neu_ex = st.session_state.df_financial_phrasebank[
                st.session_state.df_financial_phrasebank["sentiment"] == "neutral"
            ]["text"].iloc[0]
            st.markdown(f"Example Positive: `{pos_ex}`")
            st.markdown(f"Example Negative: `{neg_ex}`")
            st.markdown(f"Example Neutral: `{neu_ex}`")
        except Exception:
            st.warning(
                "Could not display example sentences (dataset may be empty or schema may differ).")

        st.subheader("Real-world: SEC 10‑K Risk Factor Excerpts (Unlabeled)")
        st.write(
            f"Extracted {len(st.session_state.df_10k)} 10‑K risk-factor paragraphs.")
        st.dataframe(st.session_state.df_10k.head())

        if len(st.session_state.df_10k) > 0:
            st.markdown(
                f"Example 10‑K paragraph: `{st.session_state.df_10k['text'].iloc[0]}`")

        with st.expander("Why these two datasets are complementary", expanded=False):
            st.markdown(
                """
- PhraseBank lets you **measure** out-of-sample performance (what error types the model makes).
- 10‑K risk factors let you **audit realism** (legalistic language, boilerplate, hedging, asymmetric tone).
- Your goal is not “best score”; it’s “best decision support under constraints.”
"""
            )

        with st.expander("Checkpoint (1 minute)", expanded=False):
            st.markdown(
                "**Question:** If 'neutral' is 60% of the benchmark, which metric becomes more informative than accuracy?")
            st.radio("Your answer", [
                     "Only overall sample size", "Macro‑F1", "Only accuracy"], key="cp_data")
            if st.session_state.get("cp_data"):
                if st.session_state.cp_data == "Macro‑F1":
                    st.success("✓ Correct! Macro-F1 gives equal weight to each class, preventing the majority 'neutral' class from dominating the metric. This reveals performance on minority classes (positive/negative) which are often more decision-relevant.")
                else:
                    st.info("Consider: With 60% neutral, a model could achieve 60% accuracy by always predicting 'neutral'. Macro-F1 averages F1 across all classes equally, revealing true performance on each sentiment type.")
    else:
        st.info("Load the data to proceed.")

elif st.session_state.page == "2. Text Preprocessing":
    learning_header(
        "2. Domain‑Aware Preprocessing (Input Audit)",
        "How preprocessing can change meaning in finance—especially negation/comparatives—and why you must audit what is removed.",
        "A single dropped token (“not”) can flip your conclusion; preprocessing is part of your model risk, not a ‘cleaning detail.’",
    )

    st.markdown(
        """
Sarah understands that generic NLP preprocessing, while useful, often falls short in the nuanced world of finance. For instance, removing common "stop words" like "not" or "below" in standard NLP can completely invert the sentiment of a financial statement (e.g., "profit did **not** increase"). To avoid such critical misinterpretations, she needs a specialized preprocessing pipeline that is **domain-aware**. This pipeline will ensure that crucial financial context and negation words are preserved, setting a robust foundation for all subsequent sentiment models.
"""
    )

    if not st.session_state.data_loaded:
        st.info("Load financial data first on '1. Data Acquisition & Review'.")
    else:
        st.subheader(
            "Configure preprocessing (with finance meaning preserved)")
        st.session_state.remove_stopwords_checkbox = st.checkbox(
            "Remove common words (keeps negation + comparatives important in finance)",
            value=st.session_state.remove_stopwords_checkbox,
            help="We aim to remove high-frequency words that rarely change meaning, while keeping words like 'not', 'without', 'below', 'above' that can flip risk interpretation.",
        )

        if st.button("Apply preprocessing", disabled=st.session_state.preprocessing_done):
            with st.spinner("Applying preprocessing..."):
                df_fpb_copy = st.session_state.df_financial_phrasebank.copy()
                df_10k_copy = st.session_state.df_10k.copy()

                df_fpb_copy["processed_text"] = df_fpb_copy["text"].apply(
                    lambda x: " ".join(preprocess_text(
                        x, remove_stopwords=st.session_state.remove_stopwords_checkbox))
                )
                df_10k_copy["processed_text"] = df_10k_copy["text"].apply(
                    lambda x: " ".join(preprocess_text(
                        x, remove_stopwords=st.session_state.remove_stopwords_checkbox))
                )

                st.session_state.df_financial_phrasebank_processed = df_fpb_copy
                st.session_state.df_10k_processed = df_10k_copy
                st.session_state.preprocessing_done = True
                st.success("Preprocessing complete.")

        if st.session_state.preprocessing_done:
            st.subheader("Before vs after (audit)")
            fpb_raw = st.session_state.df_financial_phrasebank["text"].iloc[0]
            fpb_proc = st.session_state.df_financial_phrasebank_processed["processed_text"].iloc[0]
            tenk_raw = st.session_state.df_10k["text"].iloc[0]
            tenk_proc = st.session_state.df_10k_processed["processed_text"].iloc[0]

            st.markdown(f"**Original (PhraseBank):** `{fpb_raw}`")
            st.markdown(f"**Processed (PhraseBank):** `{fpb_proc}`")
            st.markdown("---")
            st.markdown(f"**Original (10‑K):** `{tenk_raw}`")
            st.markdown(f"**Processed (10‑K):** `{tenk_proc}`")

            # Token audit for the first benchmark example
            raw_tokens = preprocess_text(fpb_raw, remove_stopwords=False)
            proc_tokens = preprocess_text(
                fpb_raw, remove_stopwords=st.session_state.remove_stopwords_checkbox)
            removed = [t for t in raw_tokens if t not in proc_tokens]
            kept = proc_tokens[:20]

            with st.expander("Token audit (what got removed/kept on the example)", expanded=False):
                st.markdown("**Removed tokens (sample):** " +
                            (", ".join(removed[:30]) if removed else "_None_"))
                st.markdown("**Kept tokens (first 20):** " +
                            (", ".join(kept) if kept else "_None_"))

            st.warning(
                "Watch-out: preprocessing must be stable across time. If rules change, historical backtests may not be comparable."
            )

            with st.expander("Checkpoint (1 minute)", expanded=False):
                st.markdown(
                    "**Question:** What is the key failure if preprocessing removes 'not'?")
                st.radio("Your answer", ["Model runs faster", "No effect on meaning",
                         "Sentiment can flip sign (false positives)"], key="cp_prep")
                if st.session_state.get("cp_prep"):
                    if st.session_state.cp_prep == "Sentiment can flip sign (false positives)":
                        st.success("✓ Correct! Removing 'not' turns 'profits did NOT increase' into 'profits increase', completely inverting the sentiment. This is a critical preprocessing error in finance where negation changes investment implications.")
                    else:
                        st.info("Consider: 'Margins improved' vs 'Margins did NOT improve' have opposite implications. Removing negation words like 'not', 'without', 'below' can flip your signal's sign entirely.")
        else:
            st.info("Apply preprocessing to proceed.")

elif st.session_state.page == "3. LM Lexicon Model":
    learning_header(
        "3. LM Dictionary Baseline (Transparent Rules)",
        "How a lexicon score is computed and why interpretability comes from ‘showing the words’ that drive the score.",
        "As a first-pass risk scan, you want something fast and explainable—then you validate where it fails.",
    )

    st.markdown(
        """
Sarah begins with the Loughran-McDonald (LM) financial sentiment dictionary, a standard tool developed specifically for financial text analysis. Unlike generic sentiment dictionaries (e.g., VADER, TextBlob) that might misclassify words like "liability," "tax," or "cost" as negative (when they are often neutral in a financial context), the LM dictionary is precisely tailored to the nuances of corporate disclosures. This approach offers high interpretability and is quick to implement, providing an immediate, albeit sometimes simplistic, sentiment score.
"""
    )

    st.subheader("Mathematical Formulation")
    # Keep existing formulas verbatim
    st.markdown(
        r"The LM sentiment score $S_{LM}(d)$ for a document $d$ is calculated as the normalized difference between the count of positive words $N_{pos}(d)$ and negative words $N_{neg}(d)$, divided by the total word count $N_{total}(d)$: "
    )
    st.markdown(
        r"""
$$
S_{LM}(d) = \frac{N_{pos}(d) - N_{neg}(d)}{N_{total}(d)}
$$
""")
    st.markdown(
        r"where $N_{pos}(d)$ is the count of positive LM words in document $d$, $N_{neg}(d)$ is the count of negative LM words, and $N_{total}(d)$ is the total word count (to avoid division by zero, $N_{total}(d)$ is at least 1)."
    )
    st.markdown(
        r"The score $S_{LM}(d)$ typically ranges from -1 (strongly negative) to +1 (strongly positive), with values near 0 indicating neutral or mixed sentiment. This lexicon-based method provides transparency, as Sarah can see exactly which words contribute to the sentiment score."
    )

    assumptions_box(
        [
            "Lexicon methods assume sentiment is the sum of word-level evidence; they do not ‘understand’ sentence logic.",
            "Class mapping requires an explicit threshold τ; show it and justify it to avoid over-interpreting tiny scores.",
        ],
        title="Interpretation guardrails",
    )

    # Thresholds (explicit) for class mapping: keep consistent with source.py
    # source.py uses sign-based mapping in lm_sentiment_score; we expose that here.
    tau = 0.0
    with st.expander("How LM score maps to a class in this app", expanded=False):
        st.markdown(
            f"""
- If score > {tau:.2f} → **positive**
- If score < -{tau:.2f} → **negative**
- Else → **neutral**
"""
        )
        st.markdown(
            "Note: if you introduce a non-zero τ later, document the rationale (noise suppression).")

    if not st.session_state.preprocessing_done:
        st.info("Complete '2. Text Preprocessing' first.")
    else:
        if st.button("Run LM sentiment analysis", disabled=st.session_state.lm_model_run):
            with st.spinner("Applying LM dictionary..."):
                df_fpb_with_lm = st.session_state.df_financial_phrasebank_processed.copy()
                df_10k_with_lm = st.session_state.df_10k_processed.copy()

                df_fpb_with_lm["lm_score"] = df_fpb_with_lm["processed_text"].apply(
                    lambda x: lm_sentiment_score(
                        x.split(), lm_positive, lm_negative)[0]
                )
                df_fpb_with_lm["lm_pred"] = df_fpb_with_lm["processed_text"].apply(
                    lambda x: lm_sentiment_score(
                        x.split(), lm_positive, lm_negative)[4]
                )

                df_10k_with_lm["lm_score"] = df_10k_with_lm["processed_text"].apply(
                    lambda x: lm_sentiment_score(
                        x.split(), lm_positive, lm_negative)[0]
                )
                df_10k_with_lm["lm_pred"] = df_10k_with_lm["processed_text"].apply(
                    lambda x: lm_sentiment_score(
                        x.split(), lm_positive, lm_negative)[4]
                )

                st.session_state.df_financial_phrasebank_processed = df_fpb_with_lm
                st.session_state.df_10k_processed = df_10k_with_lm
                st.session_state.lm_model_run = True
                st.success("LM analysis complete.")

        if st.session_state.lm_model_run:
            st.subheader("Benchmark examples (PhraseBank)")
            st.dataframe(
                st.session_state.df_financial_phrasebank_processed[[
                    "text", "sentiment", "lm_score", "lm_pred"]].head()
            )

            st.subheader("Real-world audit (10‑K): show evidence words")
            for i in range(min(3, len(st.session_state.df_10k_processed))):
                raw = st.session_state.df_10k_processed["text"].iloc[i]
                tokens = st.session_state.df_10k_processed["processed_text"].iloc[i].split(
                )
                score, n_pos, n_neg, n_total, pred = lm_sentiment_score(
                    tokens, lm_positive, lm_negative)

                pos_hits = [t for t in tokens if t in lm_positive]
                neg_hits = [t for t in tokens if t in lm_negative]
                pos_top = [w for w, c in Counter(pos_hits).most_common(8)]
                neg_top = [w for w, c in Counter(neg_hits).most_common(8)]

                st.markdown(f"**Paragraph {i+1}:** `{raw}`")
                st.markdown(
                    f"**LM score:** `{score:.4f}` (n_pos={n_pos}, n_neg={n_neg}, n_total={n_total}) → **{pred}**")
                st.markdown(
                    f"**Evidence (positive words):** `{', '.join(pos_top) if pos_top else '—'}`")
                st.markdown(
                    f"**Evidence (negative words):** `{', '.join(neg_top) if neg_top else '—'}`")
                st.markdown("---")

            with st.expander("Checkpoint (1 minute)", expanded=False):
                st.markdown(
                    "**Question:** Which LM error is most likely in 10‑K risk factors?")
                st.radio("Your answer", ["No dependence on wording", "Perfect contextual understanding",
                         "Over‑labeling as negative due to legal cautionary terms"], key="cp_lm")
                if st.session_state.get("cp_lm"):
                    if st.session_state.cp_lm == "Over‑labeling as negative due to legal cautionary terms":
                        st.success("✓ Correct! 10-K risk factors are legally required to use cautious language ('may', 'could', 'risk', 'adverse') even for standard business operations. LM dictionaries count these words without understanding legal context vs real concern.")
                    else:
                        st.info("Consider: Legal boilerplate in 10-Ks uses words like 'risk', 'adverse', 'harm' as required disclosures, not necessarily signals of deteriorating fundamentals. Simple word-counting methods can't distinguish legal tone from economic risk.")
        else:
            st.info("Run the LM model to see results.")

elif st.session_state.page == "4. TF-IDF + LogReg Model":
    learning_header(
        "4. TF‑IDF + Logistic Regression (Measured + Partly Interpretable)",
        "How TF‑IDF turns text into features and how a linear model learns class evidence you can inspect (top-weighted terms).",
        "This is a practical ‘middle ground’: stronger than lexicons, still auditable enough for model risk discussions.",
    )

    st.markdown(
        """
While the LM dictionary is interpretable, Sarah knows its rule-based nature can be rigid. To achieve higher accuracy and capture more nuanced patterns, she turns to a traditional machine learning approach: TF-IDF for text vectorization combined with Logistic Regression for classification. This method learns sentiment patterns directly from the labeled Financial PhraseBank dataset, allowing it to identify important words and even combinations of words (bigrams like "not profitable") that predict sentiment. This is a step towards data-driven intelligence, moving beyond fixed lexicons.
"""
    )

    st.subheader("Mathematical Formulation")
    # Keep existing formulas verbatim
    st.markdown(
        r"The TF-IDF (Term Frequency-Inverse Document Frequency) value for a word $w$ in document $d$ is given by:")
    st.markdown(r"""
$$
TF-IDF(w, d) = TF(w, d) \times IDF(w)
$$
""")
    st.markdown(
        r"where $TF(w, d)$ (Term Frequency) is the number of times word $w$ appears in document $d$, often normalized:")
    st.markdown(
        r"""
$$
TF(w, d) = \frac{\text{count of word w in document d}}{\text{|document d|}}
$$
""")
    st.markdown(
        r"and $IDF(w)$ (Inverse Document Frequency) measures how much information the word provides:")
    st.markdown(r"""
$$
IDF(w) = \log \frac{N}{1 + |\{d : w \in d\}|}
$$
""")
    st.markdown(
        r"where $N$ is the total number of documents, and $|\{d : w \in d\}|$ is the number of documents containing word $w$. TF-IDF effectively up-weights rare but informative words and down-weights common, less informative words. Including an $ngram\_range=(1,2)$ in the vectorizer allows capturing bigrams, which are crucial for detecting negation patterns (e.g., ""not profitable"")."
    )
    st.markdown(
        r"Logistic Regression then models the probability of a document belonging to a certain sentiment class based on these TF-IDF features. For a multinomial case (like negative, neutral, positive), the probability of class $k$ given a document's TF-IDF vector $\text{tfidf}(d)$ is:")
    st.markdown(
        r"""
$$
P(y = k | d) = \frac{\exp(\beta_k^T \text{tfidf}(d))}{\sum_j \exp(\beta_j^T \text{tfidf}(d))}
$$
""")
    st.markdown(
        r"where $\beta_k$ represents the learned coefficient vector for class $k$. The magnitude and sign of these coefficients reveal which words (or bigrams) are most predictive of each sentiment class."
    )

    assumptions_box(
        [
            "Evaluation is done on a held-out test set (out-of-sample estimate).",
            "Macro‑F1 is emphasized because classes are imbalanced (neutral often dominates).",
            "Top-weight terms are associations in the training set; they are not causal drivers.",
        ]
    )

    if not st.session_state.preprocessing_done:
        st.info("Complete '2. Text Preprocessing' first.")
    else:
        if st.button("Train TF‑IDF + Logistic Regression (with holdout test set)", disabled=st.session_state.tfidf_logreg_trained):
            with st.spinner("Training model..."):
                (
                    tfidf_vectorizer_obj,
                    lr_model_obj,
                    X_train_raw,
                    X_test_raw,
                    y_train,
                    y_test,
                    feature_names,
                    sentiment_classes,
                ) = train_tfidf_logreg_model(st.session_state.df_financial_phrasebank_processed)

                st.session_state.tfidf_vectorizer = tfidf_vectorizer_obj
                st.session_state.lr_model = lr_model_obj
                st.session_state.X_train_fpb = X_train_raw
                st.session_state.X_test_fpb = X_test_raw
                st.session_state.y_train_fpb = y_train
                st.session_state.y_test_fpb = y_test
                st.session_state.lr_feature_names = feature_names
                st.session_state.lr_sentiment_classes = sentiment_classes

                # Extract top predictive words per class
                lr_top_words = {"positive": [], "negative": [], "neutral": []}
                for i, cls in enumerate(sentiment_classes):
                    top_10_idx = lr_model_obj.coef_[i].argsort()[-10:][::-1]
                    lr_top_words[str(cls)] = [feature_names[j]
                                              for j in top_10_idx]
                st.session_state.lr_top_words = lr_top_words

                st.session_state.tfidf_logreg_trained = True
                st.success("Training complete.")

        if st.session_state.tfidf_logreg_trained:
            st.subheader(
                "Benchmark performance (held-out PhraseBank test set)")
            # Need processed test texts for prediction: transform from processed_text in the processed df
            # We align by index: X_test_fpb contains raw texts from source split
            # Build a temp df to preprocess those raw texts in the same way
            tmp = pd.DataFrame({"text": st.session_state.X_test_fpb})
            tmp["processed_text"] = tmp["text"].apply(
                lambda x: " ".join(preprocess_text(
                    x, remove_stopwords=st.session_state.remove_stopwords_checkbox))
            )
            X_test_tfidf = st.session_state.tfidf_vectorizer.transform(
                tmp["processed_text"])
            y_pred_tfidf = st.session_state.lr_model.predict(X_test_tfidf)

            st.markdown("**Classification report (interpret per class):**")
            st.text(classification_report(
                st.session_state.y_test_fpb, y_pred_tfidf, zero_division=0))

            with st.expander("How to read the report (CFA lens)", expanded=False):
                st.markdown(
                    """
- **Precision (negative):** when the model flags negative tone, how often is it correct (review load)?
- **Recall (negative):** how many truly negative texts it catches (miss risk language = costly).
- **Macro‑F1:** equal weight to each class; helps when neutral dominates.
"""
                )

            st.subheader(
                "Model evidence: top-weighted terms (associations, not causality)")
            for cls in st.session_state.lr_sentiment_classes:
                cls_str = str(cls)
                st.markdown(
                    f"**Top terms for '{cls_str}':** `{', '.join(st.session_state.lr_top_words.get(cls_str, []))}`")

            st.subheader("Apply to 10‑K risk factors (workflow realism)")
            st.warning(
                "Domain-shift guardrail: PhraseBank sentences ≠ 10‑K paragraphs. Treat predictions as a screening tool; audit examples before relying on them."
            )

            if st.button("Predict TF‑IDF + LogReg on 10‑K"):
                with st.spinner("Predicting on 10‑K paragraphs..."):
                    df_10k_with_tfidf = predict_tfidf_logreg(
                        st.session_state.df_10k_processed.copy(),
                        "processed_text",
                        st.session_state.tfidf_vectorizer,
                        st.session_state.lr_model,
                    )
                    st.session_state.df_10k_processed = df_10k_with_tfidf
                    st.success("10‑K predictions complete.")

            if st.session_state.df_10k_processed is not None and "tfidf_pred" in st.session_state.df_10k_processed.columns:
                for i in range(min(3, len(st.session_state.df_10k_processed))):
                    st.markdown(
                        f"**Paragraph {i+1}:** `{st.session_state.df_10k_processed['text'].iloc[i]}`")
                    st.markdown(
                        f"**TF‑IDF+LogReg prediction:** `{st.session_state.df_10k_processed['tfidf_pred'].iloc[i]}`")
                    st.markdown("---")

            st.markdown(
                "**Micro-example:** The bigram `not profitable` should push negative more strongly than `profitable` alone—this is what n-grams capture.")

            with st.expander("Checkpoint (1 minute)", expanded=False):
                st.markdown(
                    "**Question:** If your goal is risk monitoring, which error is usually worse?")
                st.radio("Your answer", ["Negative labeled as neutral (false negative)",                         "Neutral labeled as negative (false positive)",
                         "No errors matter equally", "Negative labeled as neutral (false negative)"], key="cp_tfidf")
                if st.session_state.cp_tfidf == "Negative labeled as neutral (false negative)":
                    st.success("✓ Correct! For risk monitoring, missing real risk language (false negative) is costly—you fail to flag deteriorating credit, operational issues, or regulatory threats. False positives just increase review workload.")
                else:
                    st.info("Consider: In risk monitoring, the cost of missing a real problem (false negative) usually exceeds the cost of over-reviewing (false positive). Adjust your threshold or weighting to prioritize recall on the 'negative' class.")
        else:
            st.info("Train the TF‑IDF + LogReg model to proceed.")

elif st.session_state.page == "5. FinBERT Transformer Model":
    learning_header(
        "5. FinBERT (Contextual Understanding)",
        "How transformers use self-attention to interpret words in context—and how to test whether ‘context’ actually improves decision usefulness.",
        "FinBERT can reduce certain failure modes (negation, nuance), but you still need example-driven audits and guardrails against over-trust.",
    )

    st.markdown(
        """
For the most advanced and context-sensitive sentiment analysis, Sarah moves to transformer models, specifically FinBERT. FinBERT is a BERT-based model pre-trained on a massive financial corpus and then fine-tuned on financial sentiment data like the Financial PhraseBank. This means it doesn't just count words or identify bigrams; it understands the semantic meaning of words based on their surrounding context. For example, "beats estimates" implies positive sentiment, whereas "market beats retreat" implies negative sentiment. This contextual understanding is what truly sets transformers apart.
"""
    )

    st.subheader("Mathematical Formulation")
    # Keep existing formulas verbatim
    st.markdown(
        r"The core of transformer models is the **Self-Attention Mechanism**. For a sequence of input tokens $x_1, \dots, x_n$, the attention score between tokens $i$ and $j$ is determined by Query ($Q$), Key ($K$), and Value ($V$) matrices derived from the input embeddings $X$:"
    )
    st.markdown(
        r"""
$$
Attention(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$
""")
    st.markdown(
        r"where $Q = XW_Q$, $K = XW_K$, $V = XW_V$ are linear projections of the input embeddings $X$ by learned weight matrices $W_Q, W_K, W_V$, and $d_k$ is the dimension of the key vectors. This allows FinBERT to ""attend"" to relevant words across the sentence, understanding relationships that simple bag-of-words models miss (e.g., connecting ""not"" to ""increase"" to correctly invert sentiment)."
    )
    st.markdown(
        r"For classification, FinBERT typically uses the representation of the special `[CLS]` token, denoted as $h_{CLS}$, which captures the aggregated meaning of the input sequence. This vector is then passed through a linear layer and a softmax function to predict the sentiment class:"
    )
    st.markdown(
        r"""
$$
P(y = k | \text{text}) = \text{softmax}(W h_{CLS} + b)_k
$$
""")
    st.markdown(
        r"where $W$ and $b$ are learned weights and bias, and $k$ represents the sentiment classes (positive, negative, neutral). This ""zero-shot transfer learning"" capability is highly valuable as Sarah can leverage a powerful, pre-trained model without needing to train it herself."
    )

    assumptions_box(
        [
            "FinBERT outputs a sentiment label; it is not a probability-calibrated risk measure.",
            "10‑K language is legalistic and may bias toward negative/neutral tone; interpret in context.",
            "Disk caching is used to keep comparisons consistent and avoid re-running expensive inference.",
        ]
    )

    if not st.session_state.preprocessing_done:
        st.info("Complete '2. Text Preprocessing' first.")
    else:
        if st.button("Initialize FinBERT (cached)", disabled=st.session_state.finbert_loaded):
            with st.spinner("Loading FinBERT model (first time may be slow)..."):
                try:
                    _ = _load_finbert_pipeline()
                    if _ is None:
                        st.error(
                            "FinBERT unavailable: transformers not installed.")
                    else:
                        st.session_state.finbert_loaded = True
                        st.success("FinBERT ready.")
                except Exception as e:
                    st.error(f"Could not initialize FinBERT: {e}")

        if st.session_state.finbert_loaded:
            st.subheader("Run FinBERT on 10‑K risk factors")
            st.session_state.finbert_batch_size = st.slider(
                "Batch size (operating cost control)",
                min_value=1,
                max_value=64,
                value=st.session_state.finbert_batch_size,
                step=1,
                help="Bigger batch sizes can be faster, but may use more memory. This is an operating constraint, not a learning objective.",
            )

            if st.button("Predict FinBERT on 10‑K"):
                with st.spinner("Running FinBERT inference (cached to disk)..."):
                    df_10k_with_finbert = st.session_state.df_10k_processed.copy()
                    preds = finbert_predict_batch(
                        df_10k_with_finbert["text"].tolist(),
                        batch_size=st.session_state.finbert_batch_size,
                    )
                    df_10k_with_finbert["finbert_pred"] = preds
                    st.session_state.df_10k_processed = df_10k_with_finbert
                    st.success("FinBERT predictions added.")

            if st.session_state.df_10k_processed is not None and "finbert_pred" in st.session_state.df_10k_processed.columns:
                st.subheader("Examples (audit for nuance)")
                for i in range(min(3, len(st.session_state.df_10k_processed))):
                    st.markdown(
                        f"**Paragraph {i+1}:** `{st.session_state.df_10k_processed['text'].iloc[i]}`")
                    st.markdown(
                        f"**FinBERT prediction:** `{st.session_state.df_10k_processed['finbert_pred'].iloc[i]}`")
                    st.markdown("---")

                st.markdown(
                    "**Micro-example:** “The company **beat** expectations” vs “the market **beat** the stock down” — same word, different sentiment depending on context."
                )

            with st.expander("Checkpoint (1 minute)", expanded=False):
                st.markdown(
                    "**Question:** What is the best way to validate the \"context advantage\"?")
                st.radio("Your answer", ["Inspect disagreement cases vs LM and read the text",
                         "Only look at model architecture", "Inspect disagreement cases vs LM and read the text", "Assume higher accuracy always"], key="cp_finbert")
                if st.session_state.cp_finbert == "Inspect disagreement cases vs LM and read the text":
                    st.success("✓ Correct! When FinBERT disagrees with simpler models, read those examples to see if the disagreement reflects genuine contextual understanding (e.g., negation, sarcasm) or spurious patterns. This builds intuition and trust.")
                else:
                    st.info("Consider: Higher benchmark scores don't guarantee better decisions on *your* text. Inspect cases where FinBERT and LM disagree—does FinBERT capture context you care about (like 'not profitable' vs 'profitable')? Manual audit builds governance.")
        else:
            st.info("Initialize FinBERT to proceed.")

elif st.session_state.page == "6. Model Comparison":
    learning_header(
        "6. Model Comparison (Error Anatomy, Not Just a Score)",
        "How to compare models using Macro‑F1 and confusion matrices—and translate error types into costs for research vs risk workflows.",
        "Choosing a model is governance + economics: which mistakes are tolerable given your role and review capacity?",
    )

    st.markdown(
        """
Having implemented three distinct sentiment analysis approaches, Sarah's next critical step is a comprehensive comparative evaluation. For AlphaQuant Investments, choosing the right model isn't just about raw accuracy; it's about understanding the trade-offs in interpretability, computational cost, and performance across different sentiment classes. The Financial PhraseBank dataset, with its labeled sentiments, is perfect for this task. Sarah will compare Accuracy, Macro-F1 score (crucial for imbalanced datasets), and per-class F1 scores. Visualizing confusion matrices will give her an intuitive understanding of where each model succeeds or fails, guiding her decision on which tool to deploy for specific investment use cases.
"""
    )

    st.subheader("Mathematical Formulation")
    # Keep existing formulas verbatim
    st.markdown(
        "For imbalanced datasets, like the Financial PhraseBank which has a higher proportion of 'neutral' sentences, Accuracy alone can be misleading. **Macro-F1 score** is preferred because it calculates the F1-score for each class independently and then averages them, giving equal weight to each class regardless of its frequency. This prevents the score from being inflated by good performance on the majority class and highlights weaknesses in detecting minority classes (like 'negative' or 'positive'). The F1-score for class $k$ is defined as:"
    )
    st.markdown(r"""
$$
F1_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}
$$
""")
    st.markdown(
        r"where $P_k$ is Precision for class $k$ and $R_k$ is Recall for class $k$.")
    st.markdown(r"The Macro-F1 score is then:")
    st.markdown(r"""
$$
Macro-F1 = \frac{1}{K} \sum_{k=1}^K F1_k
$$
""")
    st.markdown(r"where $K$ is the number of classes.")

    assumptions_box(
        [
            "Apples-to-apples: evaluate each model on the same held-out benchmark set.",
            "Interpret confusion matrices in role terms: risk prefers high negative recall; research may prefer stable rankings.",
        ]
    )

    ready = st.session_state.lm_model_run and st.session_state.tfidf_logreg_trained and st.session_state.finbert_loaded
    if not ready:
        st.info("Prerequisites: Run LM, train TF‑IDF, and initialize FinBERT first.")
    else:
        if st.button("Evaluate all models (same test set)", disabled=st.session_state.models_evaluated):
            with st.spinner("Evaluating..."):
                # Build test set (raw texts + labels) from session state
                X_test_raw = st.session_state.X_test_fpb
                y_test = st.session_state.y_test_fpb

                # TF‑IDF predictions
                tmp = pd.DataFrame({"text": X_test_raw})
                tmp["processed_text"] = tmp["text"].apply(
                    lambda x: " ".join(preprocess_text(
                        x, remove_stopwords=st.session_state.remove_stopwords_checkbox))
                )
                X_test_tfidf = st.session_state.tfidf_vectorizer.transform(
                    tmp["processed_text"])
                y_pred_tfidf = st.session_state.lr_model.predict(X_test_tfidf)

                # FinBERT predictions
                y_pred_finbert = finbert_predict_batch(
                    X_test_raw.tolist(), batch_size=st.session_state.finbert_batch_size)

                # LM predictions (compute directly on processed tokens for test texts)
                y_pred_lm = []
                for t in X_test_raw.tolist():
                    toks = preprocess_text(
                        t, remove_stopwords=st.session_state.remove_stopwords_checkbox)
                    y_pred_lm.append(lm_sentiment_score(
                        toks, lm_positive, lm_negative)[4])

                comparison_df = pd.DataFrame(
                    {
                        "text": X_test_raw,
                        "actual": y_test,
                        "lm_pred": y_pred_lm,
                        "tfidf_pred": y_pred_tfidf,
                        "finbert_pred": y_pred_finbert,
                    }
                )

                models_to_eval = {
                    "LM Dictionary": comparison_df["lm_pred"],
                    "TF‑IDF + LogReg": comparison_df["tfidf_pred"],
                    "FinBERT": comparison_df["finbert_pred"],
                }

                perf_table, fig = evaluate_all_models(
                    comparison_df, y_test, models_to_eval)

                st.session_state.performance_table = perf_table
                st.session_state.confusion_matrices_fig = fig
                st.session_state.comparison_df = comparison_df
                st.session_state.models_evaluated = True
                st.success("Evaluation complete.")

        if st.session_state.models_evaluated:
            st.subheader("Comparative performance (PhraseBank test set)")
            st.dataframe(st.session_state.performance_table.round(3))

            st.subheader("Confusion matrices (error anatomy)")
            st.pyplot(st.session_state.confusion_matrices_fig)
            plt.close(st.session_state.confusion_matrices_fig)

            with st.expander("Decision translation (CFA/risk lens)", expanded=True):
                st.markdown(
                    """
- If **negative → neutral** errors are high: you may **miss risk language** (costly in risk monitoring).
- If **neutral → negative** errors are high: you increase review burden / false alarms.
- Choose the model that minimizes the **costly** errors for your workflow, not necessarily the one with the highest accuracy.
"""
                )

            st.warning(
                "Guardrail: treat model selection as conditional on use case (risk scan vs research signal vs reporting). One ‘best model’ rarely exists."
            )

            with st.expander("Checkpoint (1 minute)", expanded=False):
                st.markdown(
                    "**Question:** Which single view best reveals what kind of mistakes a model makes?")
                st.radio("Your answer", [
                         "Confusion matrix", "Only accuracy", "Only number of parameters"], key="cp_compare")
                if st.session_state.cp_compare == "Confusion matrix":
                    st.success("✓ Correct! The confusion matrix shows which classes are being confused (e.g., negative→neutral vs neutral→negative), revealing systematic failure modes. This is essential for risk assessment and choosing the right model for your use case.")
                else:
                    st.info("Consider: Accuracy is a single number that hides error patterns. Confusion matrices show *which* mistakes are made (false positives vs false negatives by class), enabling you to pick models based on tolerable error types.")
elif st.session_state.page == "7. Sentiment-Return Correlation":
    learning_header(
        "7. How You’d Test a Sentiment Signal (Toy Example)",
        "How to translate sentiment into a finance-native signal test: Spearman correlation + quintile spread (Q5−Q1).",
        "Signal research is about robustness: even small correlations may matter only if they survive costs, regimes, and governance.",
    )

    st.markdown(
        """
The ultimate question for Sarah is whether these sentiment scores can actually translate into an informational edge and potential "alpha" for AlphaQuant Investments. She's not just interested in model accuracy; she wants to know if high sentiment predicts higher future returns and low sentiment predicts lower returns. To conceptualize this, Sarah will perform a **sentiment-return correlation analysis** and a **quintile spread analysis**. She will simulate a dataset of financial news headlines with associated FinBERT sentiment scores and subsequent daily stock returns. This allows her to test the hypothesis that positive news sentiment can be a precursor to positive future returns, and vice-versa.
"""
    )

    st.error(
        "Important: this page uses **simulated data** to demonstrate methodology only. Do NOT interpret numbers as evidence of real-world alpha."
    )

    st.subheader("Mathematical Formulation")
    st.markdown(
        """
While sentiment-return correlations are often small (e.g., Spearman correlations of 0.02-0.08), research shows that when applied systematically across a large universe of stocks, they can be economically meaningful and contribute to risk-adjusted alpha. The quintile spread analysis, comparing the returns of the most-positive sentiment stocks to the most-negative sentiment stocks, is a standard test for identifying potential long/short opportunities.
"""
    )
    # Keep existing repeated Macro-F1 block verbatim (even though conceptually odd here)
    st.markdown(
        f"The Macro-F1 score, crucial for imbalanced datasets, is calculated as:")
    st.markdown(r"""
$$
Macro-F1 = \frac{1}{K} \sum_{k=1}^K F1_k
$$
""")
    st.markdown(r"where $F1_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$.")

    assumptions_box(
        [
            "Simulated returns illustrate testing mechanics; they exclude transaction costs and market impact.",
            "Spearman correlation is rank-based (monotonic), not linear dependence.",
            "Quintile spread (Q5−Q1) is an investable framing but needs turnover/cost analysis to be meaningful.",
        ]
    )

    if st.button("Simulate & analyze correlation", disabled=st.session_state.correlation_done):
        with st.spinner("Simulating and analyzing..."):
            (
                corr_obj,
                p_val_obj,
                annualized_quintile_returns_bps_obj,
                long_short_spread_bps_obj,
                quintile_plot_fig_obj,
            ) = run_sentiment_return_correlation_analysis()
            st.session_state.corr = corr_obj
            st.session_state.p_val = p_val_obj
            st.session_state.annualized_quintile_returns_bps = annualized_quintile_returns_bps_obj
            st.session_state.long_short_spread_bps = long_short_spread_bps_obj
            st.session_state.quintile_plot_fig = quintile_plot_fig_obj
            st.session_state.correlation_done = True
            st.success("Analysis complete.")

    if st.session_state.correlation_done:
        st.subheader("Results (simulated)")
        st.markdown(
            f"Spearman correlation (avg sentiment vs next-day return): `{st.session_state.corr:.4f}`")
        st.markdown(f"P-value: `{st.session_state.p_val:.4f}`")

        st.subheader("Annualized return by sentiment quintile (bps)")
        st.dataframe(st.session_state.annualized_quintile_returns_bps.round(0))
        st.markdown(
            f"Long–Short Spread (Q5 − Q1): `{st.session_state.long_short_spread_bps:.0f} bps`")

        st.subheader("Quintile spread plot")
        st.pyplot(st.session_state.quintile_plot_fig)
        plt.close(st.session_state.quintile_plot_fig)

        with st.expander("Decision translation", expanded=True):
            st.markdown(
                """
- If Q5−Q1 increases and is stable across subperiods, sentiment may be a useful **component signal**.
- If correlation is small but consistent, it can still matter **at scale**—but only after costs and governance.
- Next steps (real data): out-of-sample test, lag structure, transaction costs, regime stability.
"""
            )

        with st.expander("Checkpoint (1 minute)", expanded=False):
            st.markdown(
                "**Question:** Which missing input most often turns 'statistically significant' into 'not tradable'?")
            st.radio("Your answer", ["More complex model architecture",
                     "Transaction costs/turnover", "More decimals"], key="cp_signal")
            if st.session_state.get("cp_signal"):
                if st.session_state.cp_signal == "Transaction costs/turnover":
                    st.success("✓ Correct! A sentiment signal might show statistical significance but high turnover can erode profits through commissions, spreads, and market impact. Even small correlations matter only if they survive costs at scale.")
                else:
                    st.info("Consider: Many 'significant' signals disappear after accounting for realistic trading costs. If a sentiment signal requires daily rebalancing with high turnover, transaction costs and slippage can eliminate economic value despite statistical significance.")
    else:
        st.info("Run the simulation to see outputs.")

elif st.session_state.page == "8. Custom Text Analysis":
    learning_header(
        "8. Custom Text (Disagreement Builds Intuition)",
        "How to compare models on the same text and learn *why* they disagree (evidence words, learned terms, context).",
        "In real workflows, disagreement is a review trigger: it tells you where the model is uncertain or the language is nuanced.",
    )

    st.markdown(
        """
This section allows you to input your own financial text and see how each of the three models predicts its sentiment. This is a practical application to test the models on new, unseen data.
"""
    )

    ready = st.session_state.preprocessing_done
    if not ready:
        st.warning("Complete preprocessing first (Page 2).")

    if not (st.session_state.lm_model_run and st.session_state.tfidf_logreg_trained and st.session_state.finbert_loaded):
        st.warning(
            "For full comparison, run LM, train TF‑IDF, and initialize FinBERT on their pages first.")

    custom_text = st.text_area(
        "Paste a sentence/paragraph (earnings, guidance, risk language)",
        value=st.session_state.custom_text_input,
        height=150,
        help="Try an ambiguous sentence with both good news and a risk clause to see model disagreement.",
    )
    st.session_state.custom_text_input = custom_text

    if st.button("Analyze custom text"):
        if not st.session_state.preprocessing_done:
            st.error("Please complete preprocessing first.")
        else:
            with st.spinner("Analyzing..."):
                tokens = preprocess_text(
                    custom_text, remove_stopwords=st.session_state.remove_stopwords_checkbox)
                processed_custom_text = " ".join(tokens)
                results = {"Text": custom_text}

                # LM
                if st.session_state.lm_model_run:
                    lm_score_val, n_pos, n_neg, n_total, lm_pred_val = lm_sentiment_score(
                        tokens, lm_positive, lm_negative
                    )
                    results["LM Prediction"] = lm_pred_val
                    results["LM Score"] = f"{lm_score_val:.4f}"
                    results["LM Evidence (pos words)"] = ", ".join(
                        sorted(set([t for t in tokens if t in lm_positive]))[:10]) or "—"
                    results["LM Evidence (neg words)"] = ", ".join(
                        sorted(set([t for t in tokens if t in lm_negative]))[:10]) or "—"
                    results["LM Counts"] = f"n_pos={n_pos}, n_neg={n_neg}, n_total={n_total}"
                else:
                    results["LM Prediction"] = "N/A (run LM first)"

                # TF‑IDF
                if st.session_state.tfidf_logreg_trained:
                    single_df = pd.DataFrame(
                        {"text": [custom_text], "processed_text": [processed_custom_text]})
                    df_with_pred = predict_tfidf_logreg(
                        single_df, "processed_text", st.session_state.tfidf_vectorizer, st.session_state.lr_model
                    )
                    results["TF‑IDF + LogReg Prediction"] = df_with_pred["tfidf_pred"].iloc[0]
                else:
                    results["TF‑IDF + LogReg Prediction"] = "N/A (train TF‑IDF first)"

                # FinBERT
                if st.session_state.finbert_loaded:
                    try:
                        fb = finbert_predict_batch([custom_text], batch_size=1)
                        results["FinBERT Prediction"] = fb[0]
                    except Exception as e:
                        results["FinBERT Prediction"] = f"N/A ({e})"
                else:
                    results["FinBERT Prediction"] = "N/A (initialize FinBERT first)"

                st.session_state.custom_text_results = results

    if st.session_state.custom_text_results:
        st.subheader("Results (with evidence where possible)")
        for k, v in st.session_state.custom_text_results.items():
            st.markdown(f"**{k}:** `{v}`")

        st.warning(
            "Guardrail: if models disagree, treat this as a review trigger. Read the text and decide which label aligns with economic meaning (not which model you prefer)."
        )

        with st.expander("Checkpoint (1 minute)", expanded=False):
            st.markdown(
                "**Question:** When models disagree on a mixed-tone sentence, what is the most defensible next step?")
            st.radio("Your answer", ["Read the text and inspect evidence (words/phrases) before using it", "Pick the model with highest headline score",
                     "Average the labels", "Read the text and inspect evidence (words/phrases) before using it"], key="cp_custom")
            if st.session_state.cp_custom == "Read the text and inspect evidence (words/phrases) before using it":
                st.success("✓ Correct! Model disagreement is a review trigger. Read the text, examine which words each model weighted, and decide which interpretation aligns with economic reality. Governance requires human judgment on edge cases.")
            else:
                st.info("Consider: Picking the 'best' model blindly or averaging labels ignores why they disagree. Disagreement often signals ambiguous or nuanced text—exactly when you need human review. Use models as screening tools, not oracles.")

# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
