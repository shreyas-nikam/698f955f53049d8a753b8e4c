"""
app.py
------
Streamlit app: Financial Text Sentiment Analysis (From LM Dictionary to FinBERT)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from source import (
    evaluate_all_models,
    lm_negative,
    lm_positive,
    lm_sentiment_score,
    load_initial_data,
    predict_tfidf_logreg,
    preprocess_text,
    run_sentiment_return_correlation_analysis,
    train_tfidf_logreg_model,
)

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Financial Sentiment Analysis")
st.title("QuLab: Lab 8 - Sentiment from News", )
st.divider()
# -------------------------------------------------------------------
# Session state initialization
# -------------------------------------------------------------------


def initialize_session_state():
    if "page" not in st.session_state:
        st.session_state.page = "Introduction"

    # Data loading
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df_financial_phrasebank = None
        st.session_state.df_10k = None
        st.session_state.risk_factor_paragraphs = None

    # Preprocessing
    if "preprocessing_done" not in st.session_state:
        st.session_state.preprocessing_done = False
        st.session_state.remove_stopwords_checkbox = True
        st.session_state.df_financial_phrasebank_processed = None
        st.session_state.df_10k_processed = None

    # LM
    if "lm_model_run" not in st.session_state:
        st.session_state.lm_model_run = False

    # TF-IDF + LogReg
    if "tfidf_logreg_trained" not in st.session_state:
        st.session_state.tfidf_logreg_trained = False
        st.session_state.tfidf_vectorizer = None
        st.session_state.lr_model = None
        st.session_state.X_train_fpb = None
        st.session_state.X_test_fpb = None
        st.session_state.y_train_fpb = None
        st.session_state.y_test_fpb = None
        st.session_state.lr_feature_names = None
        st.session_state.lr_sentiment_classes = None
        st.session_state.lr_top_words = {
            "positive": [], "negative": [], "neutral": []}

    # FinBERT
    if "finbert_loaded" not in st.session_state:
        st.session_state.finbert_loaded = False
        st.session_state.finbert_pipeline = None
        st.session_state.finbert_batch_size = 32

    # Model comparison
    if "models_evaluated" not in st.session_state:
        st.session_state.models_evaluated = False
        st.session_state.performance_table = None
        st.session_state.confusion_matrices_fig = None
        st.session_state.comparison_df = None

    # Sentiment-return correlation
    if "correlation_done" not in st.session_state:
        st.session_state.correlation_done = False
        st.session_state.corr = None
        st.session_state.p_val = None
        st.session_state.annualized_quintile_returns_bps = None
        st.session_state.long_short_spread_bps = None
        st.session_state.quintile_plot_fig = None

    # Custom text
    if "custom_text_input" not in st.session_state:
        st.session_state.custom_text_input = (
            "The company reported a strong increase in revenue but warned about rising costs."
        )
    if "custom_text_results" not in st.session_state:
        st.session_state.custom_text_results = None


initialize_session_state()


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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _require(condition: bool, msg: str):
    if not condition:
        st.info(msg)
        st.stop()


# -------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------
if st.session_state.page == "Introduction":
    st.title("Introduction: The Quest for Alpha in Unstructured Data")
    st.markdown(
        "As a Senior Equity Analyst at AlphaQuant Investments, Sarah constantly seeks innovative ways to generate alpha and manage risk. "
        "She knows that ~80% of financial information exists as **unstructured text**—earnings call transcripts, 10‑K filings, news articles, and analyst reports. "
        "Manually sifting through this deluge is time‑consuming and prone to human bias."
    )
    st.markdown("---")
    st.markdown(
        "This application follows Sarah’s workflow to extract investment signals from text using three approaches:")
    st.markdown(
        "1. **Loughran‑McDonald Financial Dictionary** (rules‑based, domain‑specific lexicon)")
    st.markdown(
        "2. **TF‑IDF + Logistic Regression** (learns patterns from labeled data)")
    st.markdown(
        "3. **FinBERT Transformer** (contextual understanding via pre‑training)")
    st.markdown("---")
    st.markdown(
        "You’ll compare accuracy vs. interpretability vs. compute cost, and then connect sentiment to a conceptual return signal."
    )

elif st.session_state.page == "1. Data Acquisition & Review":
    st.title("1. Laying the Foundation: Data Acquisition and Initial Review")
    st.markdown(
        "Sarah starts with two datasets:\n\n"
        "- **Financial PhraseBank** (labeled benchmark sentences for evaluation)\n"
        "- **SEC 10‑K risk factor excerpts** (real‑world, unlabeled text for application)"
    )
    st.markdown("---")

    if st.button("Load Financial Data", disabled=st.session_state.data_loaded):
        with st.spinner("Loading data..."):
            try:
                df_fpb, df_10k_raw, risk_paragraphs = load_initial_data()
                st.session_state.df_financial_phrasebank = df_fpb
                st.session_state.df_10k = df_10k_raw
                st.session_state.risk_factor_paragraphs = risk_paragraphs
                st.session_state.data_loaded = True
                st.success("Financial data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")

    if st.session_state.data_loaded:
        st.subheader("Financial PhraseBank Dataset (Labeled)")
        st.write(
            f"Dataset size: {len(st.session_state.df_financial_phrasebank)} sentences")
        st.dataframe(st.session_state.df_financial_phrasebank.head())

        st.write("Label Distribution:")
        st.dataframe(
            st.session_state.df_financial_phrasebank["sentiment"].value_counts())

        st.markdown(
            f"Example Positive Sentence: `{st.session_state.df_financial_phrasebank[st.session_state.df_financial_phrasebank['sentiment'] == 'positive']['text'].iloc[0]}`"
        )
        st.markdown(
            f"Example Negative Sentence: `{st.session_state.df_financial_phrasebank[st.session_state.df_financial_phrasebank['sentiment'] == 'negative']['text'].iloc[0]}`"
        )

        st.markdown("---")
        st.subheader("SEC 10-K Risk Factor Excerpts (Unlabeled)")
        st.write(
            f"Extracted {len(st.session_state.df_10k)} SEC 10‑K risk factor paragraphs.")
        st.dataframe(st.session_state.df_10k.head())
        st.markdown(
            f"Example 10‑K Risk Factor Paragraph: `{st.session_state.df_10k['text'].iloc[0]}`")

        st.markdown("---")
        st.subheader("Explanation of Execution")
        st.markdown(
            "Sarah now has labeled benchmark data for **quantitative evaluation** and real‑world 10‑K text for **practical application**. "
            "The PhraseBank label imbalance makes **Macro‑F1** particularly informative later."
        )
    else:
        st.info("Please load financial data to proceed.")

elif st.session_state.page == "2. Text Preprocessing":
    st.title("2. Sarah's NLP Workbench: Tailoring Text Preprocessing for Finance")
    _require(st.session_state.data_loaded,
             "Please load financial data first from '1. Data Acquisition & Review'.")

    st.markdown(
        "Generic preprocessing can break finance meaning (e.g., removing **not** flips sentiment). "
        "Sarah uses a **domain‑aware** pipeline that preserves negation/comparison terms and key symbols like **%** and **$**."
    )
    st.markdown("---")

    st.session_state.remove_stopwords_checkbox = st.checkbox(
        "Remove Stopwords (domain-aware; preserves 'not', 'down', 'below', etc.)",
        value=st.session_state.remove_stopwords_checkbox,
    )

    if st.button("Apply Preprocessing", disabled=st.session_state.preprocessing_done):
        with st.spinner("Applying preprocessing..."):
            df_fpb = st.session_state.df_financial_phrasebank.copy()
            df_10k = st.session_state.df_10k.copy()

            df_fpb["processed_text"] = df_fpb["text"].apply(
                lambda x: " ".join(preprocess_text(
                    x, remove_stopwords=st.session_state.remove_stopwords_checkbox))
            )
            df_10k["processed_text"] = df_10k["text"].apply(
                lambda x: " ".join(preprocess_text(
                    x, remove_stopwords=st.session_state.remove_stopwords_checkbox))
            )

            st.session_state.df_financial_phrasebank_processed = df_fpb
            st.session_state.df_10k_processed = df_10k
            st.session_state.preprocessing_done = True
            st.success("Text preprocessing applied!")

    if st.session_state.preprocessing_done:
        st.subheader("Preprocessed Text Examples")
        st.markdown(
            f"**Original (PhraseBank):** `{st.session_state.df_financial_phrasebank['text'].iloc[0]}`")
        st.markdown(
            f"**Processed (PhraseBank):** `{st.session_state.df_financial_phrasebank_processed['processed_text'].iloc[0]}`"
        )
        st.markdown("")
        st.markdown(
            f"**Original (10‑K):** `{st.session_state.df_10k['text'].iloc[0]}`")
        st.markdown(
            f"**Processed (10‑K):** `{st.session_state.df_10k_processed['processed_text'].iloc[0]}`"
        )

        st.markdown("---")
        st.subheader("Explanation of Execution")
        st.markdown(
            "The preprocessing step reduces noise while preserving finance‑critical context. "
            "This directly reduces avoidable sentiment errors later—especially around negation."
        )

elif st.session_state.page == "3. LM Lexicon Model":
    st.title("3. Approach A: The Time‑Tested Loughran‑McDonald Lexicon")
    _require(st.session_state.preprocessing_done,
             "Please complete '2. Text Preprocessing' first.")

    st.markdown(
        "The LM dictionary is a finance‑specific lexicon. It’s fast and transparent: "
        "Sarah can see exactly which words drive the score."
    )
    st.markdown("---")

    st.subheader("Mathematical Formulation")
    st.markdown(r"LM sentiment score for document $d$:")
    st.markdown(
        r"$$S_{LM}(d) = \frac{N_{pos}(d) - N_{neg}(d)}{N_{total}(d)}$$")

    if st.button("Run LM Sentiment Analysis", disabled=st.session_state.lm_model_run):
        with st.spinner("Applying Loughran‑McDonald lexicon..."):
            df_fpb = st.session_state.df_financial_phrasebank_processed.copy()
            df_10k = st.session_state.df_10k_processed.copy()

            df_fpb["lm_score"] = df_fpb["processed_text"].apply(
                lambda x: lm_sentiment_score(
                    x.split(), lm_positive, lm_negative)[0]
            )
            df_fpb["lm_pred"] = df_fpb["processed_text"].apply(
                lambda x: lm_sentiment_score(
                    x.split(), lm_positive, lm_negative)[4]
            )

            df_10k["lm_score"] = df_10k["processed_text"].apply(
                lambda x: lm_sentiment_score(
                    x.split(), lm_positive, lm_negative)[0]
            )
            df_10k["lm_pred"] = df_10k["processed_text"].apply(
                lambda x: lm_sentiment_score(
                    x.split(), lm_positive, lm_negative)[4]
            )

            st.session_state.df_financial_phrasebank_processed = df_fpb
            st.session_state.df_10k_processed = df_10k
            st.session_state.lm_model_run = True
            st.success("LM Sentiment Analysis completed!")

    if st.session_state.lm_model_run:
        st.subheader("LM Sentiment Examples (Financial PhraseBank)")
        st.dataframe(st.session_state.df_financial_phrasebank_processed[[
                     "text", "sentiment", "lm_score", "lm_pred"]].head())

        st.subheader("LM Sentiment Examples (SEC 10‑K Risk Factors)")
        for i in range(min(3, len(st.session_state.df_10k_processed))):
            st.markdown(
                f"**Paragraph {i+1}:** `{st.session_state.df_10k_processed['text'].iloc[i]}`")
            st.markdown(
                f"**LM Score:** `{st.session_state.df_10k_processed['lm_score'].iloc[i]:.4f}`, "
                f"**Predicted:** `{st.session_state.df_10k_processed['lm_pred'].iloc[i]}`"
            )
            st.markdown("---")

        st.subheader("Explanation of Execution")
        st.markdown(
            "LM provides a strong baseline: fast, interpretable, but limited by **context blindness** "
            "(it can miss sarcasm, scope, and subtle phrasing)."
        )

elif st.session_state.page == "4. TF-IDF + LogReg Model":
    st.title("4. Approach B: TF‑IDF + Logistic Regression")
    _require(st.session_state.preprocessing_done,
             "Please complete '2. Text Preprocessing' first.")

    st.markdown(
        "Sarah now learns patterns from labeled data. TF‑IDF captures important words/bigrams (e.g., *not profitable*), "
        "and Logistic Regression maps these features to sentiment classes."
    )
    st.markdown("---")

    st.subheader("Mathematical Formulation")
    st.markdown(r"$$TF\text{-}IDF(w,d)=TF(w,d)\times IDF(w)$$")

    if st.button("Train TF‑IDF + Logistic Regression Model", disabled=st.session_state.tfidf_logreg_trained):
        with st.spinner("Training TF‑IDF + Logistic Regression model..."):
            (
                tfidf_vec,
                lr_model,
                X_train_raw,
                X_test_raw,
                y_train,
                y_test,
                feature_names,
                sentiment_classes,
            ) = train_tfidf_logreg_model(st.session_state.df_financial_phrasebank_processed)

            st.session_state.tfidf_vectorizer = tfidf_vec
            st.session_state.lr_model = lr_model
            st.session_state.X_train_fpb = X_train_raw
            st.session_state.X_test_fpb = X_test_raw
            st.session_state.y_train_fpb = y_train
            st.session_state.y_test_fpb = y_test
            st.session_state.lr_feature_names = feature_names
            st.session_state.lr_sentiment_classes = sentiment_classes

            # Extract top words per class
            top_words = {"positive": [], "negative": [], "neutral": []}
            for i, cls in enumerate(sentiment_classes):
                top_idx = lr_model.coef_[i].argsort()[-10:][::-1]
                top_words[cls] = [feature_names[j] for j in top_idx]
            st.session_state.lr_top_words = top_words

            st.session_state.tfidf_logreg_trained = True
            st.success("TF‑IDF + Logistic Regression model trained!")

    if st.session_state.tfidf_logreg_trained:
        st.subheader("Model Performance (PhraseBank Test Set)")
        # Recompute predictions for this page display
        # Need processed test text: rebuild from processed df using X_test indices
        df_test = st.session_state.df_financial_phrasebank_processed.loc[
            st.session_state.X_test_fpb.index]
        X_test_proc = df_test["processed_text"].astype(str)
        X_test_tfidf = st.session_state.tfidf_vectorizer.transform(X_test_proc)
        y_pred = st.session_state.lr_model.predict(X_test_tfidf)
        st.text(
            "Classification Report:\n"
            + pd.DataFrame.from_dict(
                __import__("sklearn.metrics").metrics.classification_report(
                    st.session_state.y_test_fpb, y_pred, output_dict=True, zero_division=0
                )
            ).T.round(3).to_string()
        )

        st.subheader("Top Predictive Words (per class)")
        for cls in st.session_state.lr_sentiment_classes:
            st.markdown(
                f"**Top words for '{cls}':** `{', '.join(st.session_state.lr_top_words[cls])}`")

        st.subheader("TF‑IDF + LogReg Predictions on 10‑K Risk Factors")
        if st.button("Get TF‑IDF + LogReg Predictions on 10‑K"):
            with st.spinner("Predicting on 10‑K risk factors..."):
                df_10k_pred = predict_tfidf_logreg(
                    st.session_state.df_10k_processed.copy(),
                    "processed_text",
                    st.session_state.tfidf_vectorizer,
                    st.session_state.lr_model,
                )
                st.session_state.df_10k_processed = df_10k_pred
                st.success("10‑K predictions with TF‑IDF + LogReg completed!")

        if "tfidf_pred" in st.session_state.df_10k_processed.columns:
            for i in range(min(3, len(st.session_state.df_10k_processed))):
                st.markdown(
                    f"**Paragraph {i+1}:** `{st.session_state.df_10k_processed['text'].iloc[i]}`")
                st.markdown(
                    f"**Predicted:** `{st.session_state.df_10k_processed['tfidf_pred'].iloc[i]}`")
                st.markdown("---")

        st.subheader("Explanation of Execution")
        st.markdown(
            "This model often improves over LM by learning which words/bigrams matter in labeled finance text, "
            "offering a **balance** of accuracy and interpretability (via coefficients/top words)."
        )

elif st.session_state.page == "5. FinBERT Transformer Model":
    st.title("5. Approach C: FinBERT Transformer Model")
    _require(st.session_state.preprocessing_done,
             "Please complete '2. Text Preprocessing' first.")

    st.markdown(
        "FinBERT uses transformer self‑attention to interpret words in context. "
        "It can distinguish meanings like “beats estimates” vs “market beats retreat”."
    )
    st.markdown("---")

    st.subheader("Mathematical Formulation")
    st.markdown(
        r"$$Attention(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$")

    st.info(
        "Note: FinBERT predictions are pre-computed and loaded from JSON files for speed.")

    if st.button("Load FinBERT Predictions on 10‑K", disabled=st.session_state.finbert_loaded):
        with st.spinner("Loading FinBERT predictions from cached JSON..."):
            import json
            # Load predictions from the pre-computed JSON file
            json_path = "cache_finbert_preds/sec10k_51f1c8dcdb2243ae38956f02dc8171ca79152073111a10c6b789b3c23a4121a3.json"
            with open(json_path, "r", encoding="utf-8") as f:
                finbert_predictions = json.load(f)

            df_10k = st.session_state.df_10k_processed.copy()
            df_10k["finbert_pred"] = finbert_predictions
            st.session_state.df_10k_processed = df_10k
            st.session_state.finbert_loaded = True
            st.success("10‑K predictions with FinBERT loaded from cache!")

    if st.session_state.finbert_loaded:

        if "finbert_pred" in st.session_state.df_10k_processed.columns:
            for i in range(min(3, len(st.session_state.df_10k_processed))):
                st.markdown(
                    f"**Paragraph {i+1}:** `{st.session_state.df_10k_processed['text'].iloc[i]}`")
                st.markdown(
                    f"**Predicted:** `{st.session_state.df_10k_processed['finbert_pred'].iloc[i]}`")
                st.markdown("---")

        st.subheader("Explanation of Execution")
        st.markdown(
            "FinBERT typically delivers the strongest classification performance in finance text, "
            "but it is more computationally expensive—batching + caching makes it practical for larger corpora."
        )

elif st.session_state.page == "6. Model Comparison":
    st.title("6. Holistic Evaluation: Model Comparison")
    _require(
        st.session_state.lm_model_run and st.session_state.tfidf_logreg_trained and st.session_state.finbert_loaded,
        "Please ensure all models (LM, TF‑IDF+LogReg, FinBERT) have been run on their respective pages first.",
    )

    st.markdown(
        "Sarah compares models on the **same PhraseBank test set** using Accuracy, Macro‑F1, and confusion matrices."
    )
    st.markdown("---")

    st.subheader("Macro‑F1 (preferred under class imbalance)")
    st.markdown(r"$$Macro\text{-}F1=\frac{1}{K}\sum_{k=1}^{K}F1_k$$")

    if st.button("Evaluate All Models", disabled=st.session_state.models_evaluated):
        with st.spinner("Evaluating models..."):
            # TF-IDF predictions on test set
            df_test = st.session_state.df_financial_phrasebank_processed.loc[
                st.session_state.X_test_fpb.index]
            X_test_proc = df_test["processed_text"].astype(str)
            X_test_tfidf = st.session_state.tfidf_vectorizer.transform(
                X_test_proc)
            y_pred_tfidf = st.session_state.lr_model.predict(X_test_tfidf)

            # FinBERT predictions loaded from JSON cache (FULL dataset)
            # Note: The notebook evaluates FinBERT on the FULL dataset, not just test set
            # So we'll do the same for accurate comparison
            import json
            json_path = "cache_finbert_preds/phrasebank_test_9a8aa2aec01547ee5aa5a0d89fc66e35a682a7b1f6dc47eb16ba2d558dcbf06a.json"
            with open(json_path, "r", encoding="utf-8") as f:
                finbert_all = json.load(f)

            # Use full dataset for FinBERT evaluation (matching notebook behavior)
            df_full = st.session_state.df_financial_phrasebank_processed
            y_actual_full = df_full["sentiment"].tolist()

            # For test set only: LM and TF-IDF predictions
            y_pred_lm = df_test["lm_pred"].tolist()

            comparison_df = pd.DataFrame(
                {
                    "text": st.session_state.X_test_fpb,
                    "actual": st.session_state.y_test_fpb,
                    "lm_pred": y_pred_lm,
                    "tfidf_pred": y_pred_tfidf,
                }
            )

            # Separate FinBERT comparison on full dataset
            finbert_comparison_df = pd.DataFrame(
                {
                    "text": df_full["text"],
                    "actual": y_actual_full,
                    "finbert_pred": finbert_all
                }
            )

            # Evaluate models separately due to different dataset sizes
            # LM and TF-IDF on test set, FinBERT on full dataset (matching notebook)
            from sklearn.metrics import accuracy_score, f1_score, classification_report

            # Evaluate LM and TF-IDF on test set
            results = {}
            for name, preds in [("LM Dictionary", comparison_df["lm_pred"]),
                                ("TF‑IDF + LogReg", comparison_df["tfidf_pred"])]:
                acc = accuracy_score(comparison_df["actual"], preds)
                macro = f1_score(
                    comparison_df["actual"], preds, average="macro")
                report = classification_report(
                    comparison_df["actual"], preds, output_dict=True, zero_division=0)
                results[name] = {
                    "Accuracy": acc,
                    "Macro-F1": macro,
                    "F1 (Positive)": report.get("positive", {}).get("f1-score", 0.0),
                    "F1 (Negative)": report.get("negative", {}).get("f1-score", 0.0),
                    "F1 (Neutral)": report.get("neutral", {}).get("f1-score", 0.0),
                }

            # Evaluate FinBERT on full dataset
            acc_fb = accuracy_score(
                finbert_comparison_df["actual"], finbert_comparison_df["finbert_pred"])
            macro_fb = f1_score(
                finbert_comparison_df["actual"], finbert_comparison_df["finbert_pred"], average="macro")
            report_fb = classification_report(finbert_comparison_df["actual"], finbert_comparison_df["finbert_pred"],
                                              output_dict=True, zero_division=0)
            results["FinBERT"] = {
                "Accuracy": acc_fb,
                "Macro-F1": macro_fb,
                "F1 (Positive)": report_fb.get("positive", {}).get("f1-score", 0.0),
                "F1 (Negative)": report_fb.get("negative", {}).get("f1-score", 0.0),
                "F1 (Neutral)": report_fb.get("neutral", {}).get("f1-score", 0.0),
            }

            perf_table = pd.DataFrame(results).T

            # Generate confusion matrices
            from sklearn.metrics import confusion_matrix
            labels_order = ['negative', 'neutral', 'positive']
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(
                'Confusion Matrices: Three Sentiment Approaches', fontsize=14)

            # LM Dictionary
            cm = confusion_matrix(
                comparison_df["actual"], comparison_df["lm_pred"], labels=labels_order)
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], xticklabels=labels_order,
                        yticklabels=labels_order, cbar=False)
            axes[0].set_title("LM Dictionary (Test Set)")
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Actual")

            # TF-IDF + LogReg
            cm = confusion_matrix(
                comparison_df["actual"], comparison_df["tfidf_pred"], labels=labels_order)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1], xticklabels=labels_order,
                        yticklabels=labels_order, cbar=False)
            axes[1].set_title("TF‑IDF + LogReg (Test Set)")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("Actual")

            # FinBERT
            cm = confusion_matrix(finbert_comparison_df["actual"], finbert_comparison_df["finbert_pred"],
                                  labels=labels_order)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[2], xticklabels=labels_order,
                        yticklabels=labels_order, cbar=False)
            axes[2].set_title("FinBERT (Full Dataset)")
            axes[2].set_xlabel("Predicted")
            axes[2].set_ylabel("Actual")

            plt.tight_layout(rect=[0, 0, 1, 0.92])

            st.session_state.performance_table = perf_table
            st.session_state.confusion_matrices_fig = fig
            st.session_state.comparison_df = comparison_df
            st.session_state.finbert_comparison_df = finbert_comparison_df
            st.session_state.models_evaluated = True
            st.success("Model evaluation completed!")

    if st.session_state.models_evaluated:
        st.subheader("Comparative Model Performance")
        st.info("Note: LM Dictionary and TF-IDF+LogReg are evaluated on the test set, while FinBERT is evaluated on the full dataset (matching notebook methodology).")
        st.dataframe(st.session_state.performance_table.round(3))

        st.subheader("Confusion Matrices")
        st.pyplot(st.session_state.confusion_matrices_fig)
        plt.close(st.session_state.confusion_matrices_fig)

        st.subheader("Explanation of Execution")
        st.markdown(
            "FinBERT often leads on Macro‑F1 (better minority‑class detection). "
            "TF‑IDF+LogReg is a strong middle ground; LM is fastest and most transparent."
        )

elif st.session_state.page == "7. Sentiment-Return Correlation":
    st.title("7. From Text to Alpha: Sentiment‑Return Correlation (Conceptual)")
    st.markdown(
        "To connect sentiment to investable hypotheses, Sarah runs a **simulated** sentiment‑return analysis "
        "using Spearman correlation and a quintile spread (Q5 − Q1)."
    )
    st.markdown("---")

    if st.button("Simulate & Analyze Correlation", disabled=st.session_state.correlation_done):
        with st.spinner("Simulating data and analyzing correlation..."):
            corr, p_val, q_bps, spread_bps, fig = run_sentiment_return_correlation_analysis()
            st.session_state.corr = corr
            st.session_state.p_val = p_val
            st.session_state.annualized_quintile_returns_bps = q_bps
            st.session_state.long_short_spread_bps = spread_bps
            st.session_state.quintile_plot_fig = fig
            st.session_state.correlation_done = True
            st.success("Sentiment‑Return Correlation Analysis completed!")

    if st.session_state.correlation_done:
        st.subheader("Correlation (Simulated Data)")
        st.markdown(f"Spearman correlation: `{st.session_state.corr:.4f}`")
        st.markdown(f"P-value: `{st.session_state.p_val:.4f}`")

        st.subheader("Annualized Return by Sentiment Quintile (bps)")
        st.dataframe(st.session_state.annualized_quintile_returns_bps.round(
            0).to_frame("Annualized bps"))

        st.markdown(
            f"Long‑Short Spread (Q5 − Q1): `{st.session_state.long_short_spread_bps:.0f} bps`")

        st.subheader("Quintile Returns Plot")
        st.pyplot(st.session_state.quintile_plot_fig)
        plt.close(st.session_state.quintile_plot_fig)

        st.subheader("Explanation of Execution")
        st.markdown(
            "Even small correlations can matter economically at scale. "
            "A positive Q5−Q1 spread supports a long/short sentiment tilt as a potential alpha component."
        )
    else:
        st.info("Click 'Simulate & Analyze Correlation' to run the analysis.")

elif st.session_state.page == "8. Custom Text Analysis":
    st.title("8. Custom Text Analysis")
    st.markdown(
        "Input any financial text and compare model predictions side‑by‑side.")
    st.markdown("---")

    if not (st.session_state.lm_model_run and st.session_state.tfidf_logreg_trained and st.session_state.finbert_loaded):
        st.warning(
            "For full results, run LM (page 3), train TF‑IDF+LogReg (page 4), and load FinBERT (page 5) first."
        )

    custom_text = st.text_area(
        "Enter financial text here:",
        value=st.session_state.custom_text_input,
        height=150,
    )
    st.session_state.custom_text_input = custom_text

    if st.button("Analyze Custom Text"):
        _require(st.session_state.preprocessing_done,
                 "Please complete '2. Text Preprocessing' first.")

        with st.spinner("Analyzing custom text..."):
            tokens = preprocess_text(
                custom_text, remove_stopwords=st.session_state.remove_stopwords_checkbox)
            processed = " ".join(tokens)

            results = {"Text": custom_text}

            # LM
            if st.session_state.lm_model_run:
                lm_score_val, * \
                    _rest, lm_pred_val = lm_sentiment_score(
                        tokens, lm_positive, lm_negative)
                results["LM Dictionary Prediction"] = lm_pred_val
                results["LM Score"] = f"{lm_score_val:.4f}"
            else:
                results["LM Dictionary Prediction"] = "N/A (Model not run)"

            # TF-IDF
            if st.session_state.tfidf_logreg_trained:
                single_df = pd.DataFrame(
                    {"text": [custom_text], "processed_text": [processed]})
                pred_df = predict_tfidf_logreg(
                    single_df, "processed_text", st.session_state.tfidf_vectorizer, st.session_state.lr_model
                )
                results["TF‑IDF + LogReg Prediction"] = pred_df["tfidf_pred"].iloc[0]
            else:
                results["TF‑IDF + LogReg Prediction"] = "N/A (Model not trained)"

            # FinBERT - disabled (only pre-computed predictions available)
            results["FinBERT Prediction"] = "N/A (Only pre-computed predictions available)"

            st.session_state.custom_text_results = results
            st.success("Custom text analysis complete!")

    if st.session_state.custom_text_results:
        st.subheader("Results")
        for k, v in st.session_state.custom_text_results.items():
            st.markdown(f"**{k}:** `{v}`")

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
