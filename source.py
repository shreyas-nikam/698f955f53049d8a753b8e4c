import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

import warnings
from scipy.stats import spearmanr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FinancialSentimentAnalyzer:
    """
    A comprehensive class for performing financial sentiment analysis using various methods,
    including Loughran-McDonald dictionary, TF-IDF with Logistic Regression, and FinBERT.
    It also includes functionality for sentiment-return correlation analysis.
    """

    def __init__(self, finbert_device=-1):
        """
        Initializes the sentiment analyzer.

        Args:
            finbert_device (int): Device for FinBERT inference.
                                  -1 for CPU, 0 for GPU (if available).
        """
        self.df_financial_phrasebank = pd.DataFrame()
        self.df_10k = pd.DataFrame()
        self.X_train, self.X_test, self.y_train, self.y_test = pd.Series(), pd.Series(), pd.Series(), pd.Series()
        self.finbert_device = finbert_device
        self.finbert_pipeline = None
        self.tfidf_vectorizer = None
        self.lr_model = None
        self.lm_positive = None
        self.lm_negative = None

        self._download_nltk_data()
        self._load_lm_dictionaries()

    def _download_nltk_data(self):
        """Downloads necessary NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except nltk.downloader.DownloadError:
            nltk.download('stopwords', quiet=True)
        print("NLTK data checked/downloaded.")

    def _load_lm_dictionaries(self):
        """Loads Loughran-McDonald (simplified) word lists into instance variables."""
        self.lm_positive = {'achieve', 'attain', 'benefit', 'better', 'boost', 'creative', 'efficiency',
                            'enhance', 'excellent', 'exceed', 'favorable', 'gain', 'great', 'improve',
                            'innovation', 'opportunity', 'optimistic', 'outperform', 'positive', 'profit',
                            'progress', 'record', 'rebound', 'recovery', 'strength', 'strong', 'succeed',
                            'surpass', 'upgrade', 'upturn', 'advantage', 'growth', 'solid', 'stronger', 'well'}

        self.lm_negative = {'adverse', 'against', 'breakdown', 'burden', 'claim', 'closure', 'concern',
                            'decline', 'default', 'deficit', 'deteriorate', 'disappoint', 'downturn',
                            'failure', 'fall', 'fraud', 'impair', 'investigation', 'layoff', 'litigation',
                            'loss', 'negative', 'penalty', 'plunge', 'problem', 'recall', 'restructuring',
                            'risk', 'shortfall', 'slowdown', 'sued', 'terminate', 'threat', 'unable',
                            'unfavorable', 'violation', 'weak', 'worse', 'writedown', 'writeoff', 'challenging',
                            'uncertainty', 'volatility', 'lower', 'decrease', 'recession'}
        print("Loughran-McDonald dictionaries loaded.")

    def load_financial_phrasebank_dataset(self):
        """
        Loads the Financial PhraseBank dataset from Hugging Face or creates a dummy DataFrame
        if loading fails. Stores it in `self.df_financial_phrasebank`.

        Returns:
            pd.DataFrame: The loaded or dummy Financial PhraseBank DataFrame.
        """
        try:
            ds = load_dataset("descartes100/enhanced-financial-phrasebank")
            self.df_financial_phrasebank = pd.DataFrame({
                'text': ds['train']['sentence'],
                'label': ds['train']['label']
            })
            sentiment_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
            self.df_financial_phrasebank['sentiment'] = self.df_financial_phrasebank['label'].map(sentiment_to_label)
            self.df_financial_phrasebank = self.df_financial_phrasebank[['text', 'label', 'sentiment']]
            print(f"Financial PhraseBank Dataset size: {len(self.df_financial_phrasebank)} sentences")
            print("Financial PhraseBank Label distribution:\n", self.df_financial_phrasebank['sentiment'].value_counts())
        except Exception as e:
            print(f"Error loading Financial PhraseBank dataset: {type(e).__name__}: {e}")
            print("Creating a filled dummy DataFrame to prevent further errors.")
            dummy_data = {
                'text': [
                    "The company reported a record profit increase of 20% in the last quarter.",
                    "Revenue was largely flat, showing no significant change year-over-year.",
                    "Despite cost-cutting measures, the firm announced a substantial loss.",
                    "Expectations for future growth remain positive due to new market opportunities.",
                    "The market reaction was neutral following the CEO's vague statements.",
                    "Increased competition led to a sharp decline in market share."
                ],
                'label': [2, 1, 0, 2, 1, 0],
                'sentiment': ['positive', 'neutral', 'negative', 'positive', 'neutral', 'negative']
            }
            self.df_financial_phrasebank = pd.DataFrame(dummy_data)
        return self.df_financial_phrasebank

    def load_sec_10k_risk_factors(self, file_path='apple_10k_risk_factors.txt'):
        """
        Loads SEC 10-K risk factors from a specified file or creates a placeholder
        if the file is not found. Stores it in `self.df_10k`.

        Args:
            file_path (str): The path to the 10-K risk factors text file.

        Returns:
            pd.DataFrame: The DataFrame containing 10-K risk factor paragraphs.
        """
        risk_text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                risk_text = f.read()
            print(f"Loaded 10-K risk factors from '{file_path}'.")
        except FileNotFoundError:
            print(f"'{file_path}' not found. Creating a placeholder for demonstration.")
            risk_text = """
            Our business depends on the continued service of certain key employees. If we are unable to attract or retain qualified personnel, our business could be harmed.
            The global economy and capital markets are subject to periods of disruption and volatility. Adverse changes in economic conditions could negatively impact our financial condition and results of operations.
            We are subject to intense competition, which could adversely affect our business and operating results.
            Changes in effective tax rates, tax laws, and taxation of our international operations could harm our business.
            Our products and services may experience defects or performance problems, which could harm our reputation and results of operations.
            Compliance with new and existing laws and regulations could increase our costs and adversely affect our business.
            """
            # For an app.py, we typically avoid writing files unless explicitly configured.
            # The placeholder text is sufficient for in-memory processing.

        risk_factor_paragraphs = [p.strip() for p in risk_text.split('\n') if len(p.strip()) > 50]
        self.df_10k = pd.DataFrame({'text': risk_factor_paragraphs})
        print(f"Extracted {len(self.df_10k)} SEC 10-K risk factor paragraphs.")
        return self.df_10k

    @staticmethod
    def preprocess_text(text, remove_stopwords=True):
        """
        Standard NLP preprocessing for financial text, with domain-aware stop word removal.

        Args:
            text (str): The input financial text.
            remove_stopwords (bool): Whether to remove stop words (default: True).
                                     Crucial financial stop words are preserved.

        Returns:
            list: A list of preprocessed tokens.
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9%\s]", "", text)  # Keep alphanumeric and %
        tokens = word_tokenize(text)

        if remove_stopwords:
            stop_words_english = set(stopwords.words("english"))
            financial_keep = {"not", "no", "nor", "against", "down", "under", "up", "above", "below", "over"}
            stop_words_filtered = stop_words_english - financial_keep
            tokens = [t for t in tokens if t not in stop_words_filtered]
        return tokens

    def apply_preprocessing(self, df, text_column='text', processed_column='processed_text', remove_stopwords=True):
        """
        Applies the `preprocess_text` function to a specified column of a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the column containing raw text.
            processed_column (str): The name of the new column for processed text.
            remove_stopwords (bool): Whether to remove stop words during preprocessing.

        Returns:
            pd.DataFrame: The DataFrame with the new processed text column.
        """
        if df.empty:
            print(f"DataFrame for '{text_column}' is empty, skipping preprocessing.")
            df[processed_column] = pd.Series(dtype='str')
            return df

        df[processed_column] = df[text_column].apply(
            lambda x: " ".join(self.preprocess_text(x, remove_stopwords=remove_stopwords))
        )
        print(f"Preprocessing applied to '{text_column}' column, results in '{processed_column}'.")
        return df

    def prepare_financial_phrasebank_for_training(self, test_size=0.5, random_state=42):
        """
        Prepares the Financial PhraseBank data for model training and evaluation.
        Performs train-test split and stores results in `self.X_train`, `self.y_train`,
        `self.X_test`, `self.y_test`.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for random number generator for reproducibility.
        """
        if self.df_financial_phrasebank.empty:
            print("Financial PhraseBank DataFrame is empty, cannot prepare for training.")
            self.X_train, self.X_test, self.y_train, self.y_test = pd.Series([]), pd.Series([]), pd.Series([]), pd.Series([])
            return

        X = self.df_financial_phrasebank['processed_text']
        y = self.df_financial_phrasebank['sentiment']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Financial PhraseBank split into {len(self.X_train)} training and {len(self.X_test)} test samples.")

    def calculate_lm_sentiment(self, df, processed_text_column='processed_text', pos_threshold=0.02, neg_threshold=-0.02):
        """
        Applies Loughran-McDonald dictionary-based sentiment scoring to a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            processed_text_column (str): The name of the column containing preprocessed text tokens (space-separated).
            pos_threshold (float): Score threshold for 'positive' sentiment.
            neg_threshold (float): Score threshold for 'negative' sentiment.

        Returns:
            pd.DataFrame: The DataFrame with 'lm_score' and 'lm_pred' columns added.
        """
        if df.empty:
            print(f"DataFrame for LM sentiment '{processed_text_column}' is empty, skipping LM sentiment calculation.")
            df['lm_score'] = pd.Series(dtype='float64')
            df['lm_pred'] = pd.Series(dtype='object')
            return df

        def _lm_score_single(text_tokens_str):
            text_tokens = text_tokens_str.split()
            n_pos = sum(1 for t in text_tokens if t in self.lm_positive)
            n_neg = sum(1 for t in text_tokens if t in self.lm_negative)
            n_total = max(len(text_tokens), 1)
            score = (n_pos - n_neg) / n_total

            if score > pos_threshold:
                predicted_sentiment = 'positive'
            elif score < neg_threshold:
                predicted_sentiment = 'negative'
            else:
                predicted_sentiment = 'neutral'
            return score, predicted_sentiment

        lm_results = df[processed_text_column].apply(_lm_score_single)
        df['lm_score'] = [r[0] for r in lm_results]
        df['lm_pred'] = [r[1] for r in lm_results]
        print(f"LM sentiment scores calculated for '{processed_text_column}'.")
        return df

    def train_tfidf_logistic_regression(self, max_features=5000, ngram_range=(1, 2),
                                        min_df=1, max_df=1.0, C=1.0, max_iter=1000, random_state=42):
        """
        Trains a TF-IDF vectorizer and Logistic Regression model using Financial PhraseBank data.
        Stores the trained vectorizer and model in instance variables.

        Args:
            max_features (int): Maximum number of features (terms) for TF-IDF.
            ngram_range (tuple): Range of n-grams to be extracted.
            min_df (int/float): When building the vocabulary, ignore terms that have a document
                               frequency strictly lower than the given threshold.
            max_df (int/float): When building the vocabulary, ignore terms that have a document
                               frequency strictly higher than the given threshold.
            C (float): Inverse of regularization strength for Logistic Regression.
            max_iter (int): Maximum number of iterations for Logistic Regression solver.
            random_state (int): Seed for random number generator.

        Returns:
            tuple: (tfidf_vectorizer, lr_model, y_pred) - trained objects and test predictions.
        """
        if self.X_train.empty or self.X_test.empty:
            print("Training data for TF-IDF + Logistic Regression is empty, skipping training.")
            self.tfidf_vectorizer = None
            self.lr_model = None
            return None, None, []

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=min_df,
            max_df=max_df, sublinear_tf=True
        )
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)

        self.lr_model = LogisticRegression(
            C=C, max_iter=max_iter, class_weight='balanced', random_state=random_state
        )
        self.lr_model.fit(X_train_tfidf, self.y_train)
        y_pred = self.lr_model.predict(X_test_tfidf)

        print("\n--- TF-IDF + Logistic Regression Performance on Financial PhraseBank ---")
        print(classification_report(self.y_test, y_pred))

        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        sentiment_classes = self.lr_model.classes_
        print("\n--- Top Predictive Words (Unigrams/Bigrams) from Logistic Regression ---")
        for i, cls in enumerate(sentiment_classes):
            top_10_idx = self.lr_model.coef_[i].argsort()[-10:][::-1]
            bottom_10_idx = self.lr_model.coef_[i].argsort()[:10]
            top_words = [feature_names[j] for j in top_10_idx]
            bottom_words = [feature_names[j] for j in bottom_10_idx]
            print(f"Top POSITIVE words for '{cls}': {', '.join(top_words)}")
            print(f"Top NEGATIVE words for '{cls}': {', '.join(bottom_words)}")

        return self.tfidf_vectorizer, self.lr_model, y_pred

    def predict_with_tfidf_logistic_regression(self, df, text_column='processed_text'):
        """
        Applies the trained TF-IDF + Logistic Regression model to a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the column containing preprocessed text.

        Returns:
            pd.DataFrame: The DataFrame with a 'tfidf_pred' column added.
        """
        if self.lr_model is None or self.tfidf_vectorizer is None:
            print("TF-IDF + Logistic Regression model not trained. Skipping prediction.")
            df['tfidf_pred'] = 'N/A'
            return df

        if df.empty:
            print(f"DataFrame for TF-IDF prediction on '{text_column}' is empty, skipping prediction.")
            df['tfidf_pred'] = 'N/A'
            return df

        X_transformed = self.tfidf_vectorizer.transform(df[text_column])
        df['tfidf_pred'] = self.lr_model.predict(X_transformed)
        print(f"TF-IDF + LogReg predictions applied to '{text_column}'.")
        return df

    def load_finbert_pipeline(self):
        """
        Loads the FinBERT model pipeline from Hugging Face if not already loaded.

        Returns:
            transformers.pipeline: The FinBERT sentiment analysis pipeline.
        """
        if self.finbert_pipeline is None:
            print(f"Loading FinBERT model on device: {self.finbert_device}")
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=self.finbert_device
            )
        return self.finbert_pipeline

    def predict_with_finbert_batch(self, texts, batch_size=32):
        """
        Performs batch inference with FinBERT to handle large datasets efficiently.
        Truncates inputs to 512 tokens, which is a common limit for BERT-like models.

        Args:
            texts (list): A list of text strings to analyze.
            batch_size (int): The number of texts to process in each batch.

        Returns:
            list: A list of predicted sentiment labels ('negative', 'neutral', 'positive').
        """
        if self.finbert_pipeline is None:
            self.load_finbert_pipeline()

        if not texts:
            print("No texts provided for FinBERT prediction.")
            return []

        all_predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            truncated_batch = [text[:512] for text in batch_texts] # Truncate to max 512 tokens
            preds = self.finbert_pipeline(truncated_batch)
            all_predictions.extend([p['label'].lower() for p in preds])
        print(f"FinBERT predictions generated for {len(texts)} texts.")
        return all_predictions

    def evaluate_models_and_plot(self, lm_preds, tfidf_preds, finbert_preds, save_path='sentiment_comparison_confusion_matrices.png'):
        """
        Evaluates the performance of different sentiment models on the test set
        and generates confusion matrices.

        Args:
            lm_preds (list): List of sentiment predictions from LM dictionary for the test set.
            tfidf_preds (list): List of sentiment predictions from TF-IDF + LogReg for the test set.
            finbert_preds (list): List of sentiment predictions from FinBERT for the test set.
            save_path (str): File path to save the confusion matrices plot.
        """
        if self.y_test.empty or self.X_test.empty:
            print("Test data is empty, skipping model evaluation and plotting.")
            return

        # Ensure predictions are Series with the correct index for alignment
        lm_preds_series = pd.Series(lm_preds, index=self.X_test.index, name='lm_pred').fillna('neutral')
        tfidf_preds_series = pd.Series(tfidf_preds, index=self.X_test.index, name='tfidf_pred').fillna('neutral')
        finbert_preds_series = pd.Series(finbert_preds, index=self.X_test.index, name='finbert_pred').fillna('neutral')

        comparison_df = pd.DataFrame({
            'text': self.X_test,
            'actual': self.y_test,
            'lm_pred': lm_preds_series,
            'tfidf_pred': tfidf_preds_series,
            'finbert_pred': finbert_preds_series
        })

        models_to_evaluate = {
            'LM Dictionary': comparison_df['lm_pred'],
            'TF-IDF + LogReg': comparison_df['tfidf_pred'],
            'FinBERT': comparison_df['finbert_pred']
        }

        results = {}
        labels_order = ['negative', 'neutral', 'positive']

        print("\n--- Comparative Model Performance on Financial PhraseBank Test Set ---")
        for name, predictions in models_to_evaluate.items():
            # Filter out any 'N/A' or non-standard labels if they exist in predictions
            valid_indices = predictions.isin(labels_order)
            actual_valid = comparison_df['actual'][valid_indices]
            predictions_valid = predictions[valid_indices]

            if actual_valid.empty:
                print(f"No valid predictions for {name}, skipping evaluation.")
                results[name] = {'Accuracy': 0, 'Macro-F1': 0, 'F1 (Positive)': 0, 'F1 (Negative)': 0, 'F1 (Neutral)': 0}
                continue

            accuracy = accuracy_score(actual_valid, predictions_valid)
            report_dict = classification_report(actual_valid, predictions_valid, output_dict=True, zero_division=0, labels=labels_order)
            macro_f1 = report_dict['macro avg']['f1-score']

            f1_positive = report_dict['positive']['f1-score'] if 'positive' in report_dict else 0
            f1_negative = report_dict['negative']['f1-score'] if 'negative' in report_dict else 0
            f1_neutral = report_dict['neutral']['f1-score'] if 'neutral' in report_dict else 0

            results[name] = {
                'Accuracy': accuracy,
                'Macro-F1': macro_f1,
                'F1 (Positive)': f1_positive,
                'F1 (Negative)': f1_negative,
                'F1 (Neutral)': f1_neutral
            }

        performance_table = pd.DataFrame(results).T
        print(performance_table.round(3))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Confusion Matrices: Three Sentiment Approaches', fontsize=14)

        for i, (name, predictions) in enumerate(models_to_evaluate.items()):
            valid_indices = predictions.isin(labels_order)
            actual_valid = comparison_df['actual'][valid_indices]
            predictions_valid = predictions[valid_indices]
            if not actual_valid.empty:
                cm = confusion_matrix(actual_valid, predictions_valid, labels=labels_order)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                            xticklabels=labels_order, yticklabels=labels_order)
                axes[i].set_title(name)
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
            else:
                axes[i].set_title(f"{name} (No Valid Data)")
                axes[i].text(0.5, 0.5, "No valid data to plot", horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrices saved to {save_path}")
        plt.close(fig) # Close the figure to free memory

    def simulate_financial_news_and_returns(self, num_days=252, tickers=None, seed=42):
        """
        Simulates financial news headlines and next-day stock returns for correlation analysis.

        Args:
            num_days (int): Number of trading days to simulate.
            tickers (list): List of stock tickers to simulate.
            seed (int): Seed for random number generator for reproducibility.

        Returns:
            pd.DataFrame: A DataFrame with simulated news headlines, sentiment scores, and returns.
        """
        np.random.seed(seed)
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_days))
        simulated_data = []

        for date in dates:
            for ticker in tickers:
                finbert_score = np.random.uniform(-0.8, 0.8)
                base_return = np.random.normal(0, 0.005)
                next_day_return = base_return + (finbert_score * 0.001)

                simulated_data.append({
                    'date': date,
                    'ticker': ticker,
                    'headline': f"News for {ticker} on {date.strftime('%Y-%m-%d')}",
                    'finbert_score': finbert_score,
                    'next_day_return': next_day_return
                })
        print(f"Simulated {len(simulated_data)} financial news entries.")
        return pd.DataFrame(simulated_data)

    def analyze_sentiment_return_correlation(self, news_df, sentiment_column='finbert_score',
                                            return_column='next_day_return', save_path='sentiment_quintile_returns.png'):
        """
        Performs Spearman correlation and quintile analysis on sentiment and returns from a news DataFrame.

        Args:
            news_df (pd.DataFrame): DataFrame containing simulated news, sentiment scores, and returns.
            sentiment_column (str): Name of the column containing sentiment scores.
            return_column (str): Name of the column containing next-day returns.
            save_path (str): File path to save the sentiment quintile returns plot.
        """
        if news_df.empty:
            print("News DataFrame is empty, skipping correlation analysis.")
            return

        daily_sent_agg = news_df.groupby(['date', 'ticker']).agg(
            avg_sentiment=(sentiment_column, 'mean'),
            next_day_return=(return_column, 'first')
        ).reset_index()

        corr, p_val = spearmanr(daily_sent_agg['avg_sentiment'], daily_sent_agg['next_day_return'])
        print(f"\n--- Sentiment-Return Correlation Analysis (Simulated Data) ---")
        print(f"Spearman correlation (average sentiment vs next-day return): {corr:.4f}")
        print(f"P-value: {p_val:.4f}")

        # Check for sufficient unique values for quintile creation
        if daily_sent_agg['avg_sentiment'].nunique() < 5:
            print("Not enough unique sentiment scores to create 5 quintiles. Skipping quintile analysis.")
            return

        daily_sent_agg['sent_quintile'] = pd.qcut(
            daily_sent_agg['avg_sentiment'],
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates='drop' # Handle cases with many identical values gracefully
        )

        quintile_returns = daily_sent_agg.groupby('sent_quintile')['next_day_return'].mean()
        annualized_quintile_returns_bps = quintile_returns * 252 * 10000 # Assuming 252 trading days

        print("\n--- Annualized Return by Sentiment Quintile (bps) ---")
        print(annualized_quintile_returns_bps.round(0))

        if 5 in annualized_quintile_returns_bps.index and 1 in annualized_quintile_returns_bps.index:
            long_short_spread_bps = annualized_quintile_returns_bps.loc[5] - annualized_quintile_returns_bps.loc[1]
            print(f"\nLong-Short Spread (Q5 - Q1): {long_short_spread_bps:.0f} bps")
        else:
            print("\nCould not calculate Long-Short Spread (Q5 or Q1 missing, likely due to 'duplicates=drop').")

        plt.figure(figsize=(10, 6))
        annualized_quintile_returns_bps.plot(kind='bar', color='skyblue')
        plt.title('Annualized Returns by Sentiment Quintile (Simulated Data)', fontsize=14)
        plt.xlabel('Sentiment Quintile (1 = Most Negative, 5 = Most Positive)', fontsize=12)
        plt.ylabel('Annualized Return (bps)', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Sentiment quintile returns plot saved to {save_path}")
        plt.close() # Close the figure to free memory


if __name__ == "__main__":
    # Example usage when running as a script
    # For an app.py, you would typically instantiate the class and call methods as needed
    # based on user input or scheduled tasks.

    analyzer = FinancialSentimentAnalyzer(finbert_device=-1) # -1 for CPU, 0 for GPU if available

    # 1. Load and Preprocess Data
    analyzer.load_financial_phrasebank_dataset()
    analyzer.df_financial_phrasebank = analyzer.apply_preprocessing(analyzer.df_financial_phrasebank, text_column='text', processed_column='processed_text')
    analyzer.prepare_financial_phrasebank_for_training()

    analyzer.load_sec_10k_risk_factors()
    analyzer.df_10k = analyzer.apply_preprocessing(analyzer.df_10k, text_column='text', processed_column='processed_text')

    # Display example preprocessed text
    print("\nOriginal Financial PhraseBank Sentence:")
    if not analyzer.df_financial_phrasebank.empty:
        print(analyzer.df_financial_phrasebank["text"].iloc[0])
    print("\nProcessed Financial PhraseBank Sentence:")
    if not analyzer.df_financial_phrasebank.empty:
        print(analyzer.df_financial_phrasebank["processed_text"].iloc[0])
    print("\nOriginal 10-K Risk Factor Paragraph:")
    if not analyzer.df_10k.empty:
        print(analyzer.df_10k["text"].iloc[0])
    print("\nProcessed 10-K Risk Factor Paragraph:")
    if not analyzer.df_10k.empty:
        print(analyzer.df_10k["processed_text"].iloc[0])


    # 2. LM Dictionary Based Sentiment
    analyzer.df_financial_phrasebank = analyzer.calculate_lm_sentiment(analyzer.df_financial_phrasebank, processed_text_column='processed_text')
    # Get LM predictions for the test set for later comparison
    lm_preds_fpb_test = analyzer.df_financial_phrasebank.loc[analyzer.X_test.index, 'lm_pred'].fillna('neutral').tolist()

    analyzer.df_10k = analyzer.calculate_lm_sentiment(analyzer.df_10k, processed_text_column='processed_text')
    print("\n--- SEC 10-K Risk Factor LM Sentiment Examples ---")
    if not analyzer.df_10k.empty:
        for i in range(min(3, len(analyzer.df_10k))):
            print(f"Paragraph {i+1}: {analyzer.df_10k['text'].iloc[i]}")
            print(f"LM Score: {analyzer.df_10k['lm_score'].iloc[i]:.4f}, Predicted: {analyzer.df_10k['lm_pred'].iloc[i]}")


    # 3. TF-IDF + Logistic Regression
    _, _, tfidf_preds_fpb_test_np_array = analyzer.train_tfidf_logistic_regression()
    # Ensure tfidf_preds_fpb_test is a list for consistency in evaluation
    tfidf_preds_fpb_test = tfidf_preds_fpb_test_np_array.tolist() if tfidf_preds_fpb_test_np_array is not None and tfidf_preds_fpb_test_np_array.size > 0 else []

    analyzer.df_10k = analyzer.predict_with_tfidf_logistic_regression(analyzer.df_10k, text_column='processed_text')
    print("\n--- SEC 10-K Risk Factor TF-IDF + LogReg Sentiment Examples ---")
    if not analyzer.df_10k.empty:
        for i in range(min(3, len(analyzer.df_10k))):
            print(f"Paragraph {i+1}:\n{analyzer.df_10k['text'].iloc[i]}")
            print(f"Predicted: {analyzer.df_10k['tfidf_pred'].iloc[i]}\n")


    # 4. FinBERT (Hugging Face Transformer)
    analyzer.load_finbert_pipeline()
    finbert_preds_fpb_test = analyzer.predict_with_finbert_batch(analyzer.X_test.tolist() if not analyzer.X_test.empty else [])
    print("--- FinBERT Performance on Financial PhraseBank (Test Set) ---")
    if not analyzer.y_test.empty and finbert_preds_fpb_test:
        print(classification_report(analyzer.y_test, finbert_preds_fpb_test))
    else:
        print("FinBERT performance evaluation skipped due to empty test data or predictions.")

    analyzer.df_10k['finbert_pred'] = analyzer.predict_with_finbert_batch(analyzer.df_10k['text'].tolist() if not analyzer.df_10k.empty else [])
    print("\n--- SEC 10-K Risk Factor FinBERT Sentiment Examples ---")
    if not analyzer.df_10k.empty:
        for i in range(min(3, len(analyzer.df_10k))):
            print(f"Paragraph {i+1}:\n{analyzer.df_10k['text'].iloc[i]}")
            print(f"Predicted: {analyzer.df_10k['finbert_pred'].iloc[i]}\n")

    # 5. Comparative Evaluation
    if not analyzer.y_test.empty:
        analyzer.evaluate_models_and_plot(
            lm_preds_fpb_test,
            tfidf_preds_fpb_test,
            finbert_preds_fpb_test
        )
    else:
        print("Skipping comparative model evaluation as test data is empty.")

    # 6. Sentiment-Return Correlation Analysis (Simulated Data)
    simulated_news_df = analyzer.simulate_financial_news_and_returns()
    analyzer.analyze_sentiment_return_correlation(simulated_news_df)

    print("\n--- Analysis Complete ---")
    print("Financial PhraseBank DataFrame head:\n", analyzer.df_financial_phrasebank.head())
    print("\n10-K Risk Factors DataFrame head:\n", analyzer.df_10k.head())
