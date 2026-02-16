
# Streamlit Application Specification: Financial Text Sentiment Analysis

## Application Overview

This Streamlit application, "Financial Text Sentiment Analysis: From Bag-of-Words to FinBERT", is designed for CFA Charterholders and Investment Professionals like Sarah, a Senior Equity Analyst at AlphaQuant Investments. Its purpose is to demonstrate how to systematically extract investment signals from unstructured financial text using progressively sophisticated Natural Language Processing (NLP) techniques. The application guides the user through a real-world workflow, from data acquisition and preprocessing to model implementation, comparative evaluation, and conceptual correlation of sentiment with stock returns.

The application's story flow is as follows:

1.  **Introduction**: Sets the stage with Sarah's persona and the problem of deriving alpha from unstructured data.
2.  **Data Acquisition & Review**: Sarah loads and reviews benchmark financial sentiment data (Financial PhraseBank) and real-world SEC 10-K risk factor excerpts.
3.  **Text Preprocessing**: She applies a domain-aware text preprocessing pipeline, emphasizing the preservation of critical financial terms and negation.
4.  **LM Lexicon Model**: She implements a simple, interpretable Loughran-McDonald lexicon-based model, understanding its rules-based approach.
5.  **TF-IDF + Logistic Regression Model**: She then moves to a traditional machine learning model, TF-IDF with Logistic Regression, to capture more nuanced patterns from labeled data. This step includes identifying top predictive words.
6.  **FinBERT Transformer Model**: For state-of-the-art contextual understanding, she integrates the FinBERT transformer model, leveraging its pre-trained capabilities.
7.  **Model Comparison**: Sarah evaluates and compares all three models using various metrics (Accuracy, Macro-F1, Confusion Matrices) to understand their trade-offs in accuracy, interpretability, and computational cost.
8.  **Sentiment-Return Correlation**: Finally, she conceptually correlates sentiment scores with simulated historical stock returns using Spearman correlation and quintile spread analysis to explore the potential for generating alpha.
9.  **Custom Text Analysis**: A utility page for users to input their own financial text and get predictions from all three models.

This journey allows Sarah to transform qualitative assessments into systematic, data-driven insights for AlphaQuant.

## Code Requirements

The Streamlit application (`app.py`) will import functions from `source.py` and manage state using `st.session_state`.

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # For simulated data in source.py correlation
from source import (
    load_initial_data,
    preprocess_text,
    lm_positive, # Assuming these are globally available or returned by a function
    lm_negative, # from source.py upon import or dedicated function
    lm_sentiment_score, # Function from source.py
    train_tfidf_logreg_model,
    predict_tfidf_logreg,
    load_finbert_pipeline,
    finbert_predict_batch, # Function from source.py
    evaluate_all_models,
    run_sentiment_return_correlation_analysis,
    # Additional components/functions might be implicitly imported or assumed to be available
    # from the execution of the source.py module if it's not strictly functionalized.
    # For this specification, we assume the logical steps of the notebook are wrapped into functions.
    # Specifically, `lm_positive` and `lm_negative` are directly defined in source.py at a top level.
    # nltk downloads are assumed to be handled within source.py or on app startup.
    # sklearn metrics like accuracy_score, f1_score, confusion_matrix are used in evaluate_all_models
    # scipy.stats.spearmanr is used in run_sentiment_return_correlation_analysis
)

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Financial Sentiment Analysis")

# --- Session State Initialization ---
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'Introduction'
    
    # Data Loading
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df_financial_phrasebank = None
        st.session_state.df_10k = None
        st.session_state.risk_factor_paragraphs = None
    
    # Preprocessing
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
        st.session_state.remove_stopwords_checkbox = True # Default for preprocessing
        st.session_state.df_financial_phrasebank_processed = None
        st.session_state.df_10k_processed = None

    # LM Model
    if 'lm_model_run' not in st.session_state:
        st.session_state.lm_model_run = False

    # TF-IDF + LogReg Model
    if 'tfidf_logreg_trained' not in st.session_state:
        st.session_state.tfidf_logreg_trained = False
        st.session_state.tfidf_vectorizer = None
        st.session_state.lr_model = None
        st.session_state.X_train_fpb = None
        st.session_state.X_test_fpb = None
        st.session_state.y_train_fpb = None
        st.session_state.y_test_fpb = None
        st.session_state.lr_feature_names = None
        st.session_state.lr_sentiment_classes = None
        st.session_state.lr_top_words = {'positive': [], 'negative': [], 'neutral': []}

    # FinBERT Model
    if 'finbert_loaded' not in st.session_state:
        st.session_state.finbert_loaded = False
        st.session_state.finbert_pipeline = None
        st.session_state.finbert_batch_size = 32 # Default batch size

    # Model Comparison
    if 'models_evaluated' not in st.session_state:
        st.session_state.models_evaluated = False
        st.session_state.performance_table = None
        st.session_state.confusion_matrices_fig = None
        st.session_state.comparison_df = None # DataFrame containing actuals and all predictions

    # Sentiment-Return Correlation
    if 'correlation_done' not in st.session_state:
        st.session_state.correlation_done = False
        st.session_state.corr = None
        st.session_state.p_val = None
        st.session_state.annualized_quintile_returns_bps = None
        st.session_state.long_short_spread_bps = None
        st.session_state.quintile_plot_fig = None

    # Custom Text Analysis
    if 'custom_text_input' not in st.session_state:
        st.session_state.custom_text_input = "The company reported a strong increase in revenue but warned about rising costs."
    if 'custom_text_results' not in st.session_state:
        st.session_state.custom_text_results = None

# Initialize session state on first run
initialize_session_state()

# --- Sidebar Navigation ---
st.sidebar.title("Financial Sentiment Analysis")
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
        "8. Custom Text Analysis"
    ]
)

# --- Page Content Rendering ---

if st.session_state.page == "Introduction":
    st.title("Introduction: The Quest for Alpha in Unstructured Data")
    st.markdown(f"As a Senior Equity Analyst at AlphaQuant Investments, Sarah constantly seeks innovative ways to generate alpha and manage risk. She knows that approximately 80% of financial information exists as unstructured text – earnings call transcripts, 10-K filings, news articles, and analyst reports. Manually sifting through this deluge is not only time-consuming but also prone to human bias, making it impossible to scale.")
    st.markdown(f"") # Separator
    st.markdown(f"This application documents Sarah's journey to systematically extract investment signals from financial text using sentiment analysis. She will explore three progressively sophisticated natural language processing (NLP) approaches:")
    st.markdown(f"1.  **Loughran-McDonald Financial Dictionary:** A rules-based, domain-specific lexicon.")
    st.markdown(f"2.  **TF-IDF with Logistic Regression:** A traditional machine learning approach that learns patterns from labeled data.")
    st.markdown(f"3.  **FinBERT Transformer:** A state-of-the-art deep learning model pre-trained on financial text for contextual understanding.")
    st.markdown(f"") # Separator
    st.markdown(f"Sarah's goal is not just to understand these models but to apply them in a real-world workflow, evaluating their trade-offs in accuracy, interpretability, and computational cost. Ultimately, she aims to demonstrate how text sentiment can inform investment decisions, moving AlphaQuant beyond qualitative assessments to systematic, data-driven insights.")

elif st.session_state.page == "1. Data Acquisition & Review":
    st.title("1. Laying the Foundation: Data Acquisition and Initial Review")
    st.markdown(f"Sarah knows that high-quality data is the bedrock of any robust analysis. For this project, she needs both labeled data to train and evaluate her models and unlabeled, real-world financial text to apply her findings. She'll start by loading the Financial PhraseBank, a benchmark dataset for financial sentiment, and then prepare a set of SEC 10-K risk factor excerpts, which are crucial for understanding a company's potential vulnerabilities.")
    st.markdown(f"") # Separator
    st.markdown(f"The Financial PhraseBank provides sentences from financial news with expert sentiment labels, allowing her to quantitatively assess model performance. The 10-K risk factors, on the other hand, represent the kind of raw, complex, and often ambiguous text an analyst encounters daily. Applying models to this unlabeled data will demonstrate their practical utility.")

    if st.button("Load Financial Data", disabled=st.session_state.data_loaded):
        with st.spinner("Loading data... This may take a moment."):
            try:
                # Call the assumed function from source.py
                df_fpb, df_10k_raw, risk_paragraphs = load_initial_data()
                st.session_state.df_financial_phrasebank = df_fpb
                st.session_state.df_10k = df_10k_raw
                st.session_state.risk_factor_paragraphs = risk_paragraphs
                st.session_state.data_loaded = True
                st.success("Financial data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}. Please ensure 'apple_10k_risk_factors.txt' is available or generated by `source.py`.")

    if st.session_state.data_loaded:
        st.subheader("Financial PhraseBank Dataset (Labeled)")
        st.write(f"Dataset size: {len(st.session_state.df_financial_phrasebank)} sentences")
        st.dataframe(st.session_state.df_financial_phrasebank.head())
        st.write("Label Distribution:")
        st.dataframe(st.session_state.df_financial_phrasebank['sentiment'].value_counts())
        st.markdown(f"Example Positive Sentence: `{st.session_state.df_financial_phrasebank[st.session_state.df_financial_phrasebank['sentiment'] == 'positive']['text'].iloc[0]}`")
        st.markdown(f"Example Negative Sentence: `{st.session_state.df_financial_phrasebank[st.session_state.df_financial_phrasebank['sentiment'] == 'negative']['text'].iloc[0]}`")

        st.subheader("SEC 10-K Risk Factor Excerpts (Unlabeled)")
        st.write(f"Extracted {len(st.session_state.df_10k)} SEC 10-K risk factor paragraphs.")
        st.dataframe(st.session_state.df_10k.head())
        st.markdown(f"Example 10-K Risk Factor Paragraph: `{st.session_state.df_10k['text'].iloc[0]}`")
        
        st.markdown(f"---")
        st.subheader("Explanation of Execution")
        st.markdown(f"Sarah has successfully loaded the Financial PhraseBank, confirming its size and sentiment distribution. The prevalence of 'neutral' sentences (59%) highlights the need for evaluation metrics robust to class imbalance, such as Macro-F1. She has also prepared the SEC 10-K risk factor excerpts, which will serve as her real-world, unlabeled data for practical application of the sentiment models. This initial data setup is crucial for the subsequent steps of preprocessing and model training.")

    else:
        st.info("Please load financial data to proceed.")

elif st.session_state.page == "2. Text Preprocessing":
    st.title("2. Sarah's NLP Workbench: Tailoring Text Preprocessing for Finance")
    st.markdown(f"Sarah understands that generic NLP preprocessing, while useful, often falls short in the nuanced world of finance. For instance, removing common \"stop words\" like \"not\" or \"below\" in standard NLP can completely invert the sentiment of a financial statement (e.g., \"profit did **not** increase\"). To avoid such critical misinterpretations, she needs a specialized preprocessing pipeline that is **domain-aware**. This pipeline will ensure that crucial financial context and negation words are preserved, setting a robust foundation for all subsequent sentiment models.")
    st.markdown(f"---")
    
    if st.session_state.data_loaded:
        st.subheader("Configure and Apply Preprocessing")
        st.session_state.remove_stopwords_checkbox = st.checkbox(
            "Remove Stopwords (domain-aware, preserves financial terms like 'not', 'down')",
            value=st.session_state.remove_stopwords_checkbox
        )

        if st.button("Apply Preprocessing", disabled=st.session_state.preprocessing_done):
            with st.spinner("Applying preprocessing..."):
                # Create copies to store processed text
                df_fpb_copy = st.session_state.df_financial_phrasebank.copy()
                df_10k_copy = st.session_state.df_10k.copy()

                df_fpb_copy['processed_text'] = df_fpb_copy['text'].apply(
                    lambda x: " ".join(preprocess_text(x, remove_stopwords=st.session_state.remove_stopwords_checkbox))
                )
                df_10k_copy['processed_text'] = df_10k_copy['text'].apply(
                    lambda x: " ".join(preprocess_text(x, remove_stopwords=st.session_state.remove_stopwords_checkbox))
                )
                
                st.session_state.df_financial_phrasebank_processed = df_fpb_copy
                st.session_state.df_10k_processed = df_10k_copy
                st.session_state.preprocessing_done = True
                st.success("Text preprocessing applied!")

        if st.session_state.preprocessing_done:
            st.subheader("Preprocessed Text Examples")
            st.markdown(f"**Original (Financial PhraseBank):** `{st.session_state.df_financial_phrasebank['text'].iloc[0]}`")
            st.markdown(f"**Processed (Financial PhraseBank):** `{st.session_state.df_financial_phrasebank_processed['processed_text'].iloc[0]}`")
            st.markdown(f"") # Separator
            st.markdown(f"**Original (10-K Risk Factor):** `{st.session_state.df_10k['text'].iloc[0]}`")
            st.markdown(f"**Processed (10-K Risk Factor):** `{st.session_state.df_10k_processed['processed_text'].iloc[0]}`")
            
            st.markdown(f"---")
            st.subheader("Explanation of Execution")
            st.markdown(f"Sarah's custom `preprocess_text` function successfully lowercases, removes generic special characters (while preserving important financial symbols like '%'), tokenizes, and importantly, filters stop words *without* removing financially significant terms such as \"not\", \"above\", or \"below\". This ensures that crucial negation and comparative language, which directly impacts financial sentiment, is retained. The examples show the transformation from raw text to clean, tokenized strings, ready for sentiment analysis. This step is fundamental to preventing misclassification of financial statements.")

        else:
            st.info("Please apply preprocessing to see examples.")
    else:
        st.info("Please load financial data first from '1. Data Acquisition & Review' page.")

elif st.session_state.page == "3. LM Lexicon Model":
    st.title("3. Approach A: The Time-Tested Loughran-McDonald Lexicon")
    st.markdown(f"Sarah begins with the Loughran-McDonald (LM) financial sentiment dictionary, a standard tool developed specifically for financial text analysis. Unlike generic sentiment dictionaries (e.g., VADER, TextBlob) that might misclassify words like \"liability,\" \"tax,\" or \"cost\" as negative (when they are often neutral in a financial context), the LM dictionary is precisely tailored to the nuances of corporate disclosures. This approach offers high interpretability and is quick to implement, providing an immediate, albeit sometimes simplistic, sentiment score.")
    st.markdown(f"---")
    st.subheader("Mathematical Formulation")
    st.markdown(r"The LM sentiment score $S_{LM}(d)$ for a document $d$ is calculated as the normalized difference between the count of positive words $N_{pos}(d)$ and negative words $N_{neg}(d)$, divided by the total word count $N_{total}(d)$: ")
    st.markdown(r"$$S_{LM}(d) = \frac{N_{pos}(d) - N_{neg}(d)}{N_{total}(d)}$$")
    st.markdown(r"where $N_{pos}(d)$ is the count of positive LM words in document $d$, $N_{neg}(d)$ is the count of negative LM words, and $N_{total}(d)$ is the total word count (to avoid division by zero, $N_{total}(d)$ is at least 1).")
    st.markdown(r"The score $S_{LM}(d)$ typically ranges from -1 (strongly negative) to +1 (strongly positive), with values near 0 indicating neutral or mixed sentiment. This lexicon-based method provides transparency, as Sarah can see exactly which words contribute to the sentiment score.")
    st.markdown(f"---")
    
    if st.session_state.preprocessing_done:
        if st.button("Run LM Sentiment Analysis", disabled=st.session_state.lm_model_run):
            with st.spinner("Applying Loughran-McDonald lexicon..."):
                # Use copies to store LM predictions
                df_fpb_with_lm = st.session_state.df_financial_phrasebank_processed.copy()
                df_10k_with_lm = st.session_state.df_10k_processed.copy()

                # Call assumed function to apply LM sentiment (passing global LM word lists)
                # Ensure lm_positive and lm_negative are available from source.py
                df_fpb_with_lm['lm_score'] = df_fpb_with_lm['processed_text'].apply(lambda x: lm_sentiment_score(x.split(), lm_positive, lm_negative)[0])
                df_fpb_with_lm['lm_pred'] = df_fpb_with_lm['processed_text'].apply(lambda x: lm_sentiment_score(x.split(), lm_positive, lm_negative)[4])

                df_10k_with_lm['lm_score'] = df_10k_with_lm['processed_text'].apply(lambda x: lm_sentiment_score(x.split(), lm_positive, lm_negative)[0])
                df_10k_with_lm['lm_pred'] = df_10k_with_lm['processed_text'].apply(lambda x: lm_sentiment_score(x.split(), lm_positive, lm_negative)[4])

                st.session_state.df_financial_phrasebank_processed = df_fpb_with_lm
                st.session_state.df_10k_processed = df_10k_with_lm
                st.session_state.lm_model_run = True
                st.success("LM Sentiment Analysis completed!")

        if st.session_state.lm_model_run:
            st.subheader("LM Sentiment Examples (Financial PhraseBank)")
            st.dataframe(st.session_state.df_financial_phrasebank_processed[['text', 'sentiment', 'lm_score', 'lm_pred']].head())

            st.subheader("LM Sentiment Examples (SEC 10-K Risk Factors)")
            for i in range(min(3, len(st.session_state.df_10k_processed))):
                st.markdown(f"**Paragraph {i+1}:** `{st.session_state.df_10k_processed['text'].iloc[i]}`")
                st.markdown(f"**LM Score:** `{st.session_state.df_10k_processed['lm_score'].iloc[i]:.4f}`, **Predicted:** `{st.session_state.df_10k_processed['lm_pred'].iloc[i]}`")
            
            st.markdown(f"---")
            st.subheader("Explanation of Execution")
            st.markdown(f"Sarah has successfully implemented the Loughran-McDonald lexicon-based sentiment analysis. The output shows both the raw LM score and the classified sentiment ('positive', 'negative', 'neutral') for sentences from the Financial PhraseBank and paragraphs from the 10-K filing. For example, a 10-K paragraph discussing \"risk\" and \"harm\" correctly receives a negative LM score, highlighting the dictionary's ability to identify specific financial terminology. This method provides Sarah with a transparent and quick first pass at sentiment, valuable for rapidly assessing documents, but she recognizes its limitations in understanding complex sentences or contextual nuances beyond simple word counts.")
        else:
            st.info("Click 'Run LM Sentiment Analysis' to see results.")
    else:
        st.info("Please complete '2. Text Preprocessing' first.")

elif st.session_state.page == "4. TF-IDF + LogReg Model":
    st.title("4. Approach B: Enhancing Precision with TF-IDF and Logistic Regression")
    st.markdown(f"While the LM dictionary is interpretable, Sarah knows its rule-based nature can be rigid. To achieve higher accuracy and capture more nuanced patterns, she turns to a traditional machine learning approach: TF-IDF for text vectorization combined with Logistic Regression for classification. This method learns sentiment patterns directly from the labeled Financial PhraseBank dataset, allowing it to identify important words and even combinations of words (bigrams like \"not profitable\") that predict sentiment. This is a step towards data-driven intelligence, moving beyond fixed lexicons.")
    st.markdown(f"---")
    st.subheader("Mathematical Formulation")
    st.markdown(r"The TF-IDF (Term Frequency-Inverse Document Frequency) value for a word $w$ in document $d$ is given by:")
    st.markdown(r"$$TF-IDF(w, d) = TF(w, d) \times IDF(w)$$")
    st.markdown(r"where $TF(w, d)$ (Term Frequency) is the number of times word $w$ appears in document $d$, often normalized:")
    st.markdown(r"$$TF(w, d) = \frac{\text{count of word w in document d}}{\text{|document d|}}$$")
    st.markdown(r"and $IDF(w)$ (Inverse Document Frequency) measures how much information the word provides:")
    st.markdown(r"$$IDF(w) = \log \frac{N}{1 + |\{d : w \in d\}|}$$")
    st.markdown(r"where $N$ is the total number of documents, and $|\{d : w \in d\}|$ is the number of documents containing word $w$. TF-IDF effectively up-weights rare but informative words and down-weights common, less informative words. Including an $ngram\_range=(1,2)$ in the vectorizer allows capturing bigrams, which are crucial for detecting negation patterns (e.g., \"not profitable\").")
    st.markdown(r"") # Separator
    st.markdown(r"Logistic Regression then models the probability of a document belonging to a certain sentiment class based on these TF-IDF features. For a multinomial case (like negative, neutral, positive), the probability of class $k$ given a document's TF-IDF vector $\text{tfidf}(d)$ is:")
    st.markdown(r"$$P(y = k | d) = \frac{\exp(\beta_k^T \text{tfidf}(d))}{\sum_j \exp(\beta_j^T \text{tfidf}(d))}$$")
    st.markdown(r"where $\beta_k$ represents the learned coefficient vector for class $k$. The magnitude and sign of these coefficients reveal which words (or bigrams) are most predictive of each sentiment class.")
    st.markdown(f"---")

    if st.session_state.preprocessing_done:
        if st.button("Train TF-IDF + Logistic Regression Model", disabled=st.session_state.tfidf_logreg_trained):
            with st.spinner("Training TF-IDF + Logistic Regression model..."):
                # Call assumed function to train the model
                tfidf_vectorizer_obj, lr_model_obj, X_train, X_test, y_train, y_test, feature_names, sentiment_classes = \
                    train_tfidf_logreg_model(st.session_state.df_financial_phrasebank_processed)
                
                st.session_state.tfidf_vectorizer = tfidf_vectorizer_obj
                st.session_state.lr_model = lr_model_obj
                st.session_state.X_train_fpb = X_train
                st.session_state.X_test_fpb = X_test
                st.session_state.y_train_fpb = y_train
                st.session_state.y_test_fpb = y_test
                st.session_state.lr_feature_names = feature_names
                st.session_state.lr_sentiment_classes = sentiment_classes

                # Extract top predictive words
                lr_top_words = {'positive': [], 'negative': [], 'neutral': []}
                for i, cls in enumerate(sentiment_classes):
                    top_10_idx = lr_model_obj.coef_[i].argsort()[-10:][::-1]
                    bottom_10_idx = lr_model_obj.coef_[i].argsort()[:10]
                    lr_top_words[cls] = [feature_names[j] for j in top_10_idx]
                st.session_state.lr_top_words = lr_top_words

                st.session_state.tfidf_logreg_trained = True
                st.success("TF-IDF + Logistic Regression model trained!")
        
        if st.session_state.tfidf_logreg_trained:
            st.subheader("Model Training & Performance (Financial PhraseBank Test Set)")
            # Make predictions on the test set for comparison page later
            X_test_tfidf_pred = st.session_state.tfidf_vectorizer.transform(st.session_state.X_test_fpb)
            y_pred_tfidf = st.session_state.lr_model.predict(X_test_tfidf_pred)
            st.markdown(f"**Classification Report:**")
            st.text(classification_report(st.session_state.y_test_fpb, y_pred_tfidf)) # Assuming classification_report from source.py

            st.subheader("Top Predictive Words (Unigrams/Bigrams) from Logistic Regression")
            for cls in st.session_state.lr_sentiment_classes:
                st.markdown(f"**Top POSITIVE words for '{cls}':** `{', '.join(st.session_state.lr_top_words[cls])}`")

            st.subheader("TF-IDF + LogReg Sentiment Examples (SEC 10-K Risk Factors)")
            if st.button("Get TF-IDF + LogReg Predictions on 10-K"):
                with st.spinner("Predicting on 10-K risk factors..."):
                    df_10k_with_tfidf = predict_tfidf_logreg(
                        st.session_state.df_10k_processed.copy(),
                        'processed_text',
                        st.session_state.tfidf_vectorizer,
                        st.session_state.lr_model
                    )
                    st.session_state.df_10k_processed = df_10k_with_tfidf
                    st.success("10-K predictions with TF-IDF + LogReg completed!")
            
            if 'tfidf_pred' in st.session_state.df_10k_processed.columns:
                for i in range(min(3, len(st.session_state.df_10k_processed))):
                    st.markdown(f"**Paragraph {i+1}:** `{st.session_state.df_10k_processed['text'].iloc[i]}`")
                    st.markdown(f"**Predicted:** `{st.session_state.df_10k_processed['tfidf_pred'].iloc[i]}`")

            st.markdown(f"---")
            st.subheader("Explanation of Execution")
            st.markdown(f"Sarah's TF-IDF + Logistic Regression model demonstrates a more sophisticated approach. The `classification_report` provides detailed metrics for each sentiment class. Crucially, by examining the \"Top Predictive Words,\" Sarah can gain insight into the model's learned patterns. For instance, 'positive' sentiment might be driven by terms like \"growth\" or \"profit,\" while 'negative' sentiment correlates with \"loss\" or \"risk.\" The presence of bigrams such as \"not profitable\" reveals that the model effectively captures negation, a significant improvement over simple lexicon counting. Applying this to the 10-K paragraphs yields data-driven sentiment predictions that are more robust than the LM dictionary's rule-based outputs. This method offers a balance between accuracy and interpretability.")
        else:
            st.info("Click 'Train TF-IDF + Logistic Regression Model' to proceed.")
    else:
        st.info("Please complete '2. Text Preprocessing' first.")

elif st.session_state.page == "5. FinBERT Transformer Model":
    st.title("5. Approach C: The Power of Contextual Understanding with FinBERT")
    st.markdown(f"For the most advanced and context-sensitive sentiment analysis, Sarah moves to transformer models, specifically FinBERT. FinBERT is a BERT-based model pre-trained on a massive financial corpus and then fine-tuned on financial sentiment data like the Financial PhraseBank. This means it doesn't just count words or identify bigrams; it understands the semantic meaning of words based on their surrounding context. For example, \"beats estimates\" implies positive sentiment, whereas \"market beats retreat\" implies negative sentiment. This contextual understanding is what truly sets transformers apart.")
    st.markdown(f"---")
    st.subheader("Mathematical Formulation")
    st.markdown(r"The core of transformer models is the **Self-Attention Mechanism**. For a sequence of input tokens $x_1, \dots, x_n$, the attention score between tokens $i$ and $j$ is determined by Query ($Q$), Key ($K$), and Value ($V$) matrices derived from the input embeddings $X$:")
    st.markdown(r"$$Attention(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$")
    st.markdown(r"where $Q = XW_Q$, $K = XW_K$, $V = XW_V$ are linear projections of the input embeddings $X$ by learned weight matrices $W_Q, W_K, W_V$, and $d_k$ is the dimension of the key vectors. This allows FinBERT to \"attend\" to relevant words across the sentence, understanding relationships that simple bag-of-words models miss (e.g., connecting \"not\" to \"increase\" to correctly invert sentiment).")
    st.markdown(r"") # Separator
    st.markdown(r"For classification, FinBERT typically uses the representation of the special `[CLS]` token, denoted as $h_{CLS}$, which captures the aggregated meaning of the input sequence. This vector is then passed through a linear layer and a softmax function to predict the sentiment class:")
    st.markdown(r"$$P(y = k | \text{text}) = \text{softmax}(W h_{CLS} + b)_k$$")
    st.markdown(r"where $W$ and $b$ are learned weights and bias, and $k$ represents the sentiment classes (positive, negative, neutral). This \"zero-shot transfer learning\" capability is highly valuable as Sarah can leverage a powerful, pre-trained model without needing to train it herself.")
    st.markdown(f"---")

    if st.session_state.preprocessing_done:
        if st.button("Load FinBERT Model", disabled=st.session_state.finbert_loaded):
            with st.spinner("Loading FinBERT model... This may take a while the first time."):
                # Call assumed function to load the pipeline
                finbert_pipeline_obj = load_finbert_pipeline()
                st.session_state.finbert_pipeline = finbert_pipeline_obj
                st.session_state.finbert_loaded = True
                st.success("FinBERT model loaded!")

        if st.session_state.finbert_loaded:
            st.subheader("FinBERT Model Inference")
            st.session_state.finbert_batch_size = st.slider(
                "FinBERT Batch Size (for efficient processing)",
                min_value=1, max_value=64, value=st.session_state.finbert_batch_size, step=1
            )
            if st.button("Get FinBERT Predictions on 10-K"):
                with st.spinner(f"Predicting on {len(st.session_state.df_10k_processed)} 10-K risk factors with FinBERT (batch size {st.session_state.finbert_batch_size})..."):
                    # Use original text for FinBERT as it handles its own tokenization
                    df_10k_with_finbert = st.session_state.df_10k_processed.copy()
                    df_10k_with_finbert['finbert_pred'] = finbert_predict_batch(
                        df_10k_with_finbert['text'].tolist(),
                        batch_size=st.session_state.finbert_batch_size
                    )
                    st.session_state.df_10k_processed = df_10k_with_finbert
                    st.success("10-K predictions with FinBERT completed!")
            
            if 'finbert_pred' in st.session_state.df_10k_processed.columns:
                st.subheader("FinBERT Sentiment Examples (SEC 10-K Risk Factors)")
                for i in range(min(3, len(st.session_state.df_10k_processed))):
                    st.markdown(f"**Paragraph {i+1}:** `{st.session_state.df_10k_processed['text'].iloc[i]}`")
                    st.markdown(f"**Predicted:** `{st.session_state.df_10k_processed['finbert_pred'].iloc[i]}`")
            
            st.markdown(f"---")
            st.subheader("Explanation of Execution")
            st.markdown(f"FinBERT demonstrates its prowess in understanding financial sentiment with its high accuracy. The `classification_report` (often showing superior performance in general) typically shows superior performance compared to the previous methods, especially in handling the 'negative' and 'positive' classes more effectively. Sarah notes that FinBERT's predictions for the 10-K risk factors are often more nuanced, capturing subtle financial sentiment that a simple word count or even bigram analysis might miss. The batch inference function ensures that even large sets of documents can be processed efficiently. This \"zero-shot\" capability, where the model performs well without explicit user training, highlights the power of transfer learning and provides Sarah with a state-of-the-art tool for complex text analysis.")
        else:
            st.info("Click 'Load FinBERT Model' to initialize the model.")
    else:
        st.info("Please complete '2. Text Preprocessing' first.")


elif st.session_state.page == "6. Model Comparison":
    st.title("6. Holistic Evaluation: Model Comparison for Strategic Selection")
    st.markdown(f"Having implemented three distinct sentiment analysis approaches, Sarah's next critical step is a comprehensive comparative evaluation. For AlphaQuant Investments, choosing the right model isn't just about raw accuracy; it's about understanding the trade-offs in interpretability, computational cost, and performance across different sentiment classes. The Financial PhraseBank dataset, with its labeled sentiments, is perfect for this task. Sarah will compare Accuracy, Macro-F1 score (crucial for imbalanced datasets), and per-class F1 scores. Visualizing confusion matrices will give her an intuitive understanding of where each model succeeds or fails, guiding her decision on which tool to deploy for specific investment use cases.")
    st.markdown(f"---")
    st.subheader("Mathematical Formulation")
    st.markdown(f"For imbalanced datasets, like the Financial PhraseBank which has a higher proportion of 'neutral' sentences, Accuracy alone can be misleading. **Macro-F1 score** is preferred because it calculates the F1-score for each class independently and then averages them, giving equal weight to each class regardless of its frequency. This prevents the score from being inflated by good performance on the majority class and highlights weaknesses in detecting minority classes (like 'negative' or 'positive'). The F1-score for class $k$ is defined as:")
    st.markdown(r"$$F1_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$")
    st.markdown(r"where $P_k$ is Precision for class $k$ and $R_k$ is Recall for class $k$.")
    st.markdown(r"The Macro-F1 score is then:")
    st.markdown(r"$$Macro-F1 = \frac{1}{K} \sum_{k=1}^K F1_k$$")
    st.markdown(r"where $K$ is the number of classes.")
    st.markdown(f"---")

    if st.session_state.lm_model_run and st.session_state.tfidf_logreg_trained and st.session_state.finbert_loaded:
        if st.button("Evaluate All Models", disabled=st.session_state.models_evaluated):
            with st.spinner("Evaluating all models and generating comparison plots..."):
                # Re-run TF-IDF & FinBERT predictions on the *test set* of Financial PhraseBank
                # This ensures consistent comparison with the LM model which was applied to the entire FPB.
                # For `evaluate_all_models` from `source.py` we need predictions on the same `y_test`.

                # Get TF-IDF + LogReg predictions on FPB test set
                X_test_tfidf_transformed = st.session_state.tfidf_vectorizer.transform(st.session_state.X_test_fpb)
                y_pred_tfidf_fpb_test = st.session_state.lr_model.predict(X_test_tfidf_transformed)

                # Get FinBERT predictions on FPB test set
                # FinBERT uses original text, so we'll need to pass original X_test.
                # Assuming X_test_fpb in session state is still raw text.
                y_pred_finbert_fpb_test = finbert_predict_batch(
                    st.session_state.X_test_fpb.tolist(),
                    batch_size=st.session_state.finbert_batch_size
                )
                
                # Get LM predictions on FPB test set
                # LM was applied to df_financial_phrasebank_processed which contains lm_pred
                # We need to filter it for the X_test_fpb indices.
                df_fpb_test_filtered = st.session_state.df_financial_phrasebank_processed.loc[st.session_state.X_test_fpb.index]
                y_pred_lm_fpb_test = df_fpb_test_filtered['lm_pred'].tolist()

                # Create comparison DataFrame for evaluation
                comparison_df_temp = pd.DataFrame({
                    'text': st.session_state.X_test_fpb,
                    'actual': st.session_state.y_test_fpb,
                    'lm_pred': y_pred_lm_fpb_test,
                    'tfidf_pred': y_pred_tfidf_fpb_test,
                    'finbert_pred': y_pred_finbert_fpb_test
                })

                models_to_evaluate_preds = {
                    'LM Dictionary': comparison_df_temp['lm_pred'],
                    'TF-IDF + LogReg': comparison_df_temp['tfidf_pred'],
                    'FinBERT': comparison_df_temp['finbert_pred']
                }

                # Call the assumed evaluation function
                performance_table_obj, confusion_matrices_fig_obj = evaluate_all_models(
                    comparison_df_temp, st.session_state.y_test_fpb, models_to_evaluate_preds
                )
                
                st.session_state.performance_table = performance_table_obj
                st.session_state.confusion_matrices_fig = confusion_matrices_fig_obj
                st.session_state.comparison_df = comparison_df_temp # Store for potential future use
                st.session_state.models_evaluated = True
                st.success("Model evaluation completed!")

        if st.session_state.models_evaluated:
            st.subheader("Comparative Model Performance on Financial PhraseBank Test Set")
            st.dataframe(st.session_state.performance_table.round(3))

            st.subheader("Confusion Matrices: Three Sentiment Approaches")
            st.pyplot(st.session_state.confusion_matrices_fig)
            plt.close(st.session_state.confusion_matrices_fig) # Clear figure to prevent display issues

            st.markdown(f"---")
            st.subheader("Explanation of Execution")
            st.markdown(f"The comparative performance table clearly shows the trade-offs. FinBERT typically achieves the highest Macro-F1 and accuracy, particularly excelling in identifying positive and negative sentiments, which are crucial for investment decisions. The confusion matrices provide a visual diagnosis:")
            st.markdown(f"*   **Loughran-McDonald:** Often has lower true positives for positive/negative classes and more misclassifications, especially confusing neutral with other classes.")
            st.markdown(f"*   **TF-IDF + LogReg:** Shows improvement, learning to distinguish classes better, but might still struggle with subtle nuances or highly imbalanced classes.")
            st.markdown(f"*   **FinBERT:** Demonstrates strong diagonal values, indicating accurate predictions across all classes, and fewer errors in distinguishing between negative/positive and neutral.")
            st.markdown(f"") # Separator
            st.markdown(f"For Sarah, this analysis confirms that FinBERT offers the best overall performance for detecting critical investment signals. While the LM dictionary is quick and interpretable for a first pass, and TF-IDF+LogReg offers a data-driven improvement, FinBERT's contextual understanding makes it the most powerful tool for high-stakes decisions at AlphaQuant. The choice of model, however, will also depend on the specific use case's interpretability and computational requirements.")
        else:
            st.info("Please run all models on their respective pages before evaluating.")
    else:
        st.info("Please ensure all models (LM, TF-IDF+LogReg, FinBERT) have been run on their respective pages before proceeding to evaluation.")

elif st.session_state.page == "7. Sentiment-Return Correlation":
    st.title("7. From Text to Alpha: Correlating Sentiment with Stock Returns (Conceptual)")
    st.markdown(f"The ultimate question for Sarah is whether these sentiment scores can actually translate into an informational edge and potential \"alpha\" for AlphaQuant Investments. She's not just interested in model accuracy; she wants to know if high sentiment predicts higher future returns and low sentiment predicts lower returns. To conceptualize this, Sarah will perform a **sentiment-return correlation analysis** and a **quintile spread analysis**. She will simulate a dataset of financial news headlines with associated FinBERT sentiment scores and subsequent daily stock returns. This allows her to test the hypothesis that positive news sentiment can be a precursor to positive future returns, and vice-versa.")
    st.markdown(f"---")
    st.subheader("Mathematical Formulation")
    st.markdown(f"While sentiment-return correlations are often small (e.g., Spearman correlations of 0.02-0.08), research shows that when applied systematically across a large universe of stocks, they can be economically meaningful and contribute to risk-adjusted alpha. The quintile spread analysis, comparing the returns of the most-positive sentiment stocks to the most-negative sentiment stocks, is a standard test for identifying potential long/short opportunities.")
    st.markdown(f"The Macro-F1 score, crucial for imbalanced datasets, is calculated as:")
    st.markdown(r"$$Macro-F1 = \frac{1}{K} \sum_{k=1}^K F1_k$$")
    st.markdown(r"where $F1_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$.") # F1_k formula repeated for clarity in context
    st.markdown(f"---")

    if st.button("Simulate & Analyze Correlation", disabled=st.session_state.correlation_done):
        with st.spinner("Simulating data and analyzing sentiment-return correlation..."):
            # Call the assumed correlation analysis function
            corr_obj, p_val_obj, annualized_quintile_returns_bps_obj, long_short_spread_bps_obj, quintile_plot_fig_obj = \
                run_sentiment_return_correlation_analysis()
            
            st.session_state.corr = corr_obj
            st.session_state.p_val = p_val_obj
            st.session_state.annualized_quintile_returns_bps = annualized_quintile_returns_bps_obj
            st.session_state.long_short_spread_bps = long_short_spread_bps_obj
            st.session_state.quintile_plot_fig = quintile_plot_fig_obj
            st.session_state.correlation_done = True
            st.success("Sentiment-Return Correlation Analysis completed!")

    if st.session_state.correlation_done:
        st.subheader("Sentiment-Return Correlation Analysis (Simulated Data)")
        st.markdown(f"Spearman correlation (average sentiment vs next-day return): `{st.session_state.corr:.4f}`")
        st.markdown(f"P-value: `{st.session_state.p_val:.4f}`")

        st.subheader("Annualized Return by Sentiment Quintile (bps)")
        st.dataframe(st.session_state.annualized_quintile_returns_bps.round(0))
        st.markdown(f"Long-Short Spread (Q5 - Q1): `{st.session_state.long_short_spread_bps:.0f} bps`")

        st.subheader("Annualized Returns by Sentiment Quintile Plot")
        st.pyplot(st.session_state.quintile_plot_fig)
        plt.close(st.session_state.quintile_plot_fig) # Clear figure

        st.markdown(f"---")
        st.subheader("Explanation of Execution")
        st.markdown(f"Sarah's conceptual analysis demonstrates a crucial link between sentiment and returns. The Spearman correlation coefficient, even if small (as is typical for financial sentiment), indicates a directional relationship. The p-value helps assess the statistical significance of this correlation.")
        st.markdown(f"") # Separator
        st.markdown(f"More importantly, the **quintile spread analysis** reveals a potential investment signal. If the annualized return for the most positive sentiment quintile (Q5) is significantly higher than for the most negative sentiment quintile (Q1), it suggests that a strategy of \"going long\" on high-sentiment stocks and \"going short\" on low-sentiment stocks could generate alpha. The bar chart vividly illustrates this spread, showing a clear upward trend in returns as sentiment moves from negative to positive. A positive Q5 - Q1 spread (e.g., 300+ bps) would be considered economically significant for AlphaQuant. This exercise validates for Sarah that systematic sentiment analysis, despite small correlations, can be a valuable component of a quantitative investment strategy when applied at scale.")
    else:
        st.info("Click 'Simulate & Analyze Correlation' to run the analysis.")

elif st.session_state.page == "8. Custom Text Analysis":
    st.title("8. Custom Text Analysis")
    st.markdown(f"This section allows you to input your own financial text and see how each of the three models predicts its sentiment. This is a practical application to test the models on new, unseen data.")
    st.markdown(f"---")

    if not (st.session_state.lm_model_run and st.session_state.tfidf_logreg_trained and st.session_state.finbert_loaded):
        st.warning("Please ensure all models (LM, TF-IDF+LogReg, FinBERT) have been run/trained/loaded on their respective pages to enable full custom text analysis.")
    
    custom_text = st.text_area(
        "Enter financial text here:",
        value=st.session_state.custom_text_input,
        height=150
    )
    st.session_state.custom_text_input = custom_text

    if st.button("Analyze Custom Text"):
        if not st.session_state.preprocessing_done:
            st.error("Please complete '2. Text Preprocessing' first.")
        else:
            with st.spinner("Analyzing custom text..."):
                processed_custom_text_tokens = preprocess_text(custom_text, remove_stopwords=st.session_state.remove_stopwords_checkbox)
                processed_custom_text = " ".join(processed_custom_text_tokens)
                
                results = {"Text": custom_text}

                # LM Model
                if st.session_state.lm_model_run:
                    lm_score_val, _, _, _, lm_pred_val = lm_sentiment_score(processed_custom_text_tokens, lm_positive, lm_negative)
                    results["LM Dictionary Prediction"] = lm_pred_val
                    results["LM Score"] = f"{lm_score_val:.4f}"
                else:
                    results["LM Dictionary Prediction"] = "N/A (Model not run)"

                # TF-IDF + Logistic Regression Model
                if st.session_state.tfidf_logreg_trained:
                    # Need to wrap single prediction logic, or use existing predict_tfidf_logreg with a single-row df
                    single_df = pd.DataFrame({'text': [custom_text], 'processed_text': [processed_custom_text]})
                    df_with_tfidf_pred = predict_tfidf_logreg(single_df, 'processed_text', st.session_state.tfidf_vectorizer, st.session_state.lr_model)
                    results["TF-IDF + LogReg Prediction"] = df_with_tfidf_pred['tfidf_pred'].iloc[0]
                else:
                    results["TF-IDF + LogReg Prediction"] = "N/A (Model not trained)"

                # FinBERT Transformer Model
                if st.session_state.finbert_loaded:
                    finbert_preds = finbert_predict_batch([custom_text], batch_size=1)
                    results["FinBERT Prediction"] = finbert_preds[0]
                else:
                    results["FinBERT Prediction"] = "N/A (Model not loaded)"
                
                st.session_state.custom_text_results = results
                st.success("Custom text analysis complete!")

    if st.session_state.custom_text_results:
        st.subheader("Analysis Results:")
        for key, value in st.session_state.custom_text_results.items():
            st.markdown(f"**{key}:** `{value}`")

```
