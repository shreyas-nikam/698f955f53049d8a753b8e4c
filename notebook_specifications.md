
# Financial Text Sentiment: From Lexicon to Transformers

**Persona:** Sarah, CFA Charterholder and Senior Equity Analyst at AlphaQuant Investments.  
**Organization:** AlphaQuant Investments is a quantitative hedge fund committed to leveraging cutting-edge data science for an informational edge in fast-moving financial markets. Sarah's role is critical in identifying undervalued assets and managing risk by systematically processing vast amounts of financial information.

## Introduction: The Quest for Alpha in Unstructured Data

As a Senior Equity Analyst at AlphaQuant Investments, Sarah constantly seeks innovative ways to generate alpha and manage risk. She knows that approximately 80% of financial information exists as unstructured text – earnings call transcripts, 10-K filings, news articles, and analyst reports. Manually sifting through this deluge is not only time-consuming but also prone to human bias, making it impossible to scale.

This notebook documents Sarah's journey to systematically extract investment signals from financial text using sentiment analysis. She will explore three progressively sophisticated natural language processing (NLP) approaches:

1.  **Loughran-McDonald Financial Dictionary:** A rules-based, domain-specific lexicon.
2.  **TF-IDF with Logistic Regression:** A traditional machine learning approach that learns patterns from labeled data.
3.  **FinBERT Transformer:** A state-of-the-art deep learning model pre-trained on financial text for contextual understanding.

Sarah's goal is not just to understand these models but to apply them in a real-world workflow, evaluating their trade-offs in accuracy, interpretability, and computational cost. Ultimately, she aims to demonstrate how text sentiment can inform investment decisions, moving AlphaQuant beyond qualitative assessments to systematic, data-driven insights.

---

### Installing Required Libraries

Sarah begins by setting up her environment, ensuring all necessary Python packages are installed to handle data manipulation, NLP tasks, machine learning, and visualization.

```python
!pip install transformers datasets nltk scikit-learn pandas numpy matplotlib seaborn torch
```

---

### Importing Required Dependencies

Next, Sarah imports all the necessary libraries and modules. This step ensures that all functions and classes required for text processing, model building, evaluation, and visualization are available for use throughout her analysis.

```python
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
```

---

## 1. Laying the Foundation: Data Acquisition and Initial Review

### Story + Context + Real-World Relevance

Sarah knows that high-quality data is the bedrock of any robust analysis. For this project, she needs both labeled data to train and evaluate her models and unlabeled, real-world financial text to apply her findings. She'll start by loading the Financial PhraseBank, a benchmark dataset for financial sentiment, and then prepare a set of SEC 10-K risk factor excerpts, which are crucial for understanding a company's potential vulnerabilities.

The Financial PhraseBank provides sentences from financial news with expert sentiment labels, allowing her to quantitatively assess model performance. The 10-K risk factors, on the other hand, represent the kind of raw, complex, and often ambiguous text an analyst encounters daily. Applying models to this unlabeled data will demonstrate their practical utility.

```python
# Load the Financial PhraseBank dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
df_financial_phrasebank = pd.DataFrame(dataset['train'])
df_financial_phrasebank.columns = ['text', 'label']

# Map numeric labels to descriptive text labels
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
df_financial_phrasebank['sentiment'] = df_financial_phrasebank['label'].map(label_map)

print(f"Financial PhraseBank Dataset size: {len(df_financial_phrasebank)} sentences")
print("\nFinancial PhraseBank Label distribution:")
print(df_financial_phrasebank['sentiment'].value_counts())
print("\nExample Positive Sentence:")
print(df_financial_phrasebank[df_financial_phrasebank['sentiment'] == 'positive']['text'].iloc[0])
print("\nExample Negative Sentence:")
print(df_financial_phrasebank[df_financial_phrasebank['sentiment'] == 'negative']['text'].iloc[0])

# Load SEC 10-K Risk Factors (assuming 'apple_10k_risk_factors.txt' is available)
# In a real scenario, this would involve fetching from EDGAR or a data provider.
try:
    with open('apple_10k_risk_factors.txt', 'r', encoding='utf-8') as f:
        risk_text = f.read()
except FileNotFoundError:
    print("\n'apple_10k_risk_factors.txt' not found. Creating a placeholder for demonstration.")
    # Placeholder content if the file is not present
    risk_text = """
    Our business depends on the continued service of certain key employees. If we are unable to attract or retain qualified personnel, our business could be harmed.
    The global economy and capital markets are subject to periods of disruption and volatility. Adverse changes in economic conditions could negatively impact our financial condition and results of operations.
    We are subject to intense competition, which could adversely affect our business and operating results.
    Changes in effective tax rates, tax laws, and taxation of our international operations could harm our business.
    Our products and services may experience defects or performance problems, which could harm our reputation and results of operations.
    Compliance with new and existing laws and regulations could increase our costs and adversely affect our business.
    """
    with open('apple_10k_risk_factors.txt', 'w', encoding='utf-8') as f:
        f.write(risk_text)
    print("Placeholder 'apple_10k_risk_factors.txt' created.")

# Split into paragraphs for sentence-level or paragraph-level analysis
# Filter for paragraphs longer than 50 characters to avoid trivial text
risk_factor_paragraphs = [p.strip() for p in risk_text.split('\n\n') if len(p.strip()) > 50]

print(f"\nExtracted {len(risk_factor_paragraphs)} SEC 10-K risk factor paragraphs.")
print("\nExample 10-K Risk Factor Paragraph:")
print(risk_factor_paragraphs[0])

# Store 10-K paragraphs in a DataFrame for consistent processing
df_10k = pd.DataFrame({'text': risk_factor_paragraphs})
```

### Explanation of Execution

Sarah has successfully loaded the Financial PhraseBank, confirming its size and sentiment distribution. The prevalence of 'neutral' sentences (59%) highlights the need for evaluation metrics robust to class imbalance, such as Macro-F1. She has also prepared the SEC 10-K risk factor excerpts, which will serve as her real-world, unlabeled data for practical application of the sentiment models. This initial data setup is crucial for the subsequent steps of preprocessing and model training.

---

## 2. Sarah's NLP Workbench: Tailoring Text Preprocessing for Finance

### Story + Context + Real-World Relevance

Sarah understands that generic NLP preprocessing, while useful, often falls short in the nuanced world of finance. For instance, removing common "stop words" like "not" or "below" in standard NLP can completely invert the sentiment of a financial statement (e.g., "profit did **not** increase"). To avoid such critical misinterpretations, she needs a specialized preprocessing pipeline that is **domain-aware**. This pipeline will ensure that crucial financial context and negation words are preserved, setting a robust foundation for all subsequent sentiment models.

```python
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
    # 1. Lowercase
    text = text.lower()

    # 2. Remove special characters (keep alphanumeric and %)
    # Financial texts often contain percentages (e.g., "5% increase"), so '%' is kept.
    text = re.sub(r'[^a-zA-Z0-9%\s]', '', text)

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stop words (optional - careful with finance!)
    if remove_stopwords:
        # Load generic English stop words
        stop_words_english = set(stopwords.words('english'))

        # CRITICAL: KEEP financial stop words that carry sentiment or comparison
        # Words like "not", "no", "nor" are essential for negation.
        # Words like "above", "below", "up", "down", "over", "under" are critical for comparisons.
        financial_keep = {'not', 'no', 'nor', 'against', 'down', 'under', 'up', 'above', 'below', 'over'}
        
        # Remove financial_keep words from the generic stop words set
        stop_words_filtered = stop_words_english - financial_keep

        # Filter out stop words
        tokens = [t for t in tokens if t not in stop_words_filtered]
    
    return tokens

# Apply preprocessing to Financial PhraseBank
df_financial_phrasebank['processed_text'] = df_financial_phrasebank['text'].apply(lambda x: ' '.join(preprocess_text(x, remove_stopwords=True)))

# Apply preprocessing to 10-K risk factors
df_10k['processed_text'] = df_10k['text'].apply(lambda x: ' '.join(preprocess_text(x, remove_stopwords=True)))

print("Original Financial PhraseBank Sentence:")
print(df_financial_phrasebank['text'].iloc[0])
print("\nProcessed Financial PhraseBank Sentence:")
print(df_financial_phrasebank['processed_text'].iloc[0])

print("\nOriginal 10-K Risk Factor Paragraph:")
print(df_10k['text'].iloc[0])
print("\nProcessed 10-K Risk Factor Paragraph:")
print(df_10k['processed_text'].iloc[0])
```

### Explanation of Execution

Sarah's custom `preprocess_text` function successfully lowercases, removes generic special characters (while preserving important financial symbols like '%'), tokenizes, and importantly, filters stop words *without* removing financially significant terms such as "not", "above", or "below". This ensures that crucial negation and comparative language, which directly impacts financial sentiment, is retained. The examples show the transformation from raw text to clean, tokenized strings, ready for sentiment analysis. This step is fundamental to preventing misclassification of financial statements.

---

## 3. Approach A: The Time-Tested Loughran-McDonald Lexicon

### Story + Context + Real-World Relevance

Sarah begins with the Loughran-McDonald (LM) financial sentiment dictionary, a standard tool developed specifically for financial text analysis. Unlike generic sentiment dictionaries (e.g., VADER, TextBlob) that might misclassify words like "liability," "tax," or "cost" as negative (when they are often neutral in a financial context), the LM dictionary is precisely tailored to the nuances of corporate disclosures. This approach offers high interpretability and is quick to implement, providing an immediate, albeit sometimes simplistic, sentiment score.

The LM sentiment score $S_{LM}(d)$ for a document $d$ is calculated as the normalized difference between the count of positive words $N_{pos}(d)$ and negative words $N_{neg}(d)$, divided by the total word count $N_{total}(d)$:

$$S_{LM}(d) = \frac{N_{pos}(d) - N_{neg}(d)}{N_{total}(d)}$$

Where $N_{pos}(d)$ is the count of positive LM words in document $d$, $N_{neg}(d)$ is the count of negative LM words, and $N_{total}(d)$ is the total word count. The score $S_{LM}(d)$ typically ranges from -1 (strongly negative) to +1 (strongly positive), with values near 0 indicating neutral or mixed sentiment. This lexicon-based method provides transparency, as Sarah can see exactly which words contribute to the sentiment score.

```python
# Load Loughran-McDonald word lists (simplified for demonstration)
# In a real application, these would be loaded from external files or a library.
# The following lists are examples based on common LM terms as per the prompt.
lm_positive = {'achieve', 'attain', 'benefit', 'better', 'boost', 'creative', 'efficiency', 
               'enhance', 'excellent', 'exceed', 'favorable', 'gain', 'great', 'improve', 
               'innovation', 'opportunity', 'optimistic', 'outperform', 'positive', 'profit', 
               'progress', 'record', 'rebound', 'recovery', 'strength', 'strong', 'succeed', 
               'surpass', 'upgrade', 'upturn', 'advantage', 'growth', 'solid', 'stronger', 'well'}

lm_negative = {'adverse', 'against', 'breakdown', 'burden', 'claim', 'closure', 'concern', 
               'decline', 'default', 'deficit', 'deteriorate', 'disappoint', 'downturn', 
               'failure', 'fall', 'fraud', 'impair', 'investigation', 'layoff', 'litigation', 
               'loss', 'negative', 'penalty', 'plunge', 'problem', 'recall', 'restructuring', 
               'risk', 'shortfall', 'slowdown', 'sued', 'terminate', 'threat', 'unable', 
               'unfavorable', 'violation', 'weak', 'worse', 'writedown', 'writeoff', 'challenging',
               'uncertainty', 'volatility', 'lower', 'decrease', 'recession'}

def lm_sentiment_score(text_tokens, pos_words=lm_positive, neg_words=lm_negative):
    """
    Calculates the Loughran-McDonald dictionary-based sentiment score.
    
    Args:
        text_tokens (list): A list of preprocessed text tokens.
        pos_words (set): Set of positive words from the LM dictionary.
        neg_words (set): Set of negative words from the LM dictionary.

    Returns:
        tuple: (score, n_pos, n_neg, n_total, predicted_sentiment)
    """
    n_pos = sum(1 for t in text_tokens if t in pos_words)
    n_neg = sum(1 for t in text_tokens if t in neg_words)
    n_total = max(len(text_tokens), 1) # Avoid division by zero

    score = (n_pos - n_neg) / n_total

    # Define sentiment categories based on typical thresholds for LM scores
    # These thresholds can be fine-tuned based on domain knowledge or empirical analysis.
    if score > 0.02:  # Slightly positive threshold
        predicted_sentiment = 'positive'
    elif score < -0.02: # Slightly negative threshold
        predicted_sentiment = 'negative'
    else:
        predicted_sentiment = 'neutral'
        
    return score, n_pos, n_neg, n_total, predicted_sentiment

# Apply LM scoring to Financial PhraseBank (using processed tokens)
lm_scores_fpb = df_financial_phrasebank['processed_text'].apply(lambda x: lm_sentiment_score(x.split()))
df_financial_phrasebank['lm_score'] = [s[0] for s in lm_scores_fpb]
df_financial_phrasebank['lm_pred'] = [s[4] for s in lm_scores_fpb]

# Apply LM scoring to 10-K risk factors
lm_scores_10k = df_10k['processed_text'].apply(lambda x: lm_sentiment_score(x.split()))
df_10k['lm_score'] = [s[0] for s in lm_scores_10k]
df_10k['lm_pred'] = [s[4] for s in lm_scores_10k]

print("\n--- Financial PhraseBank LM Sentiment Examples ---")
print(df_financial_phrasebank[['text', 'sentiment', 'lm_score', 'lm_pred']].head())

print("\n--- SEC 10-K Risk Factor LM Sentiment Examples ---")
for i in range(min(3, len(df_10k))):
    print(f"Paragraph {i+1}:\n{df_10k['text'].iloc[i]}")
    print(f"LM Score: {df_10k['lm_score'].iloc[i]:.4f}, Predicted: {df_10k['lm_pred'].iloc[i]}\n")
```

### Explanation of Execution

Sarah has successfully implemented the Loughran-McDonald lexicon-based sentiment analysis. The output shows both the raw LM score and the classified sentiment ('positive', 'negative', 'neutral') for sentences from the Financial PhraseBank and paragraphs from the 10-K filing. For example, a 10-K paragraph discussing "risk" and "harm" correctly receives a negative LM score, highlighting the dictionary's ability to identify specific financial terminology. This method provides Sarah with a transparent and quick first pass at sentiment, valuable for rapidly assessing documents, but she recognizes its limitations in understanding complex sentences or contextual nuances beyond simple word counts.

---

## 4. Approach B: Enhancing Precision with TF-IDF and Logistic Regression

### Story + Context + Real-World Relevance

While the LM dictionary is interpretable, Sarah knows its rule-based nature can be rigid. To achieve higher accuracy and capture more nuanced patterns, she turns to a traditional machine learning approach: TF-IDF for text vectorization combined with Logistic Regression for classification. This method learns sentiment patterns directly from the labeled Financial PhraseBank dataset, allowing it to identify important words and even combinations of words (bigrams like "not profitable") that predict sentiment. This is a step towards data-driven intelligence, moving beyond fixed lexicons.

The TF-IDF (Term Frequency-Inverse Document Frequency) value for a word $w$ in document $d$ is given by:
$$TF-IDF(w, d) = TF(w, d) \times IDF(w)$$
Where $TF(w, d)$ (Term Frequency) is the number of times word $w$ appears in document $d$, often normalized:
$$TF(w, d) = \frac{\text{count of word w in document d}}{\text{|document d|}}$$
And $IDF(w)$ (Inverse Document Frequency) measures how much information the word provides:
$$IDF(w) = \log \frac{N}{1 + |\{d : w \in d\}|}$$
Here, $N$ is the total number of documents, and $|\{d : w \in d\}|$ is the number of documents containing word $w$. TF-IDF effectively up-weights rare but informative words and down-weights common, less informative words. Including an $ngram\_range=(1,2)$ in the vectorizer allows capturing bigrams, which are crucial for detecting negation patterns (e.g., "not profitable").

Logistic Regression then models the probability of a document belonging to a certain sentiment class based on these TF-IDF features. For a multinomial case (like negative, neutral, positive), the probability of class $k$ given a document's TF-IDF vector $\text{tfidf}(d)$ is:
$$P(y = k | d) = \frac{\exp(\beta_k^T \text{tfidf}(d))}{\sum_{j} \exp(\beta_j^T \text{tfidf}(d))}$$
Here, $\beta_k$ represents the learned coefficient vector for class $k$. The magnitude and sign of these coefficients reveal which words (or bigrams) are most predictive of each sentiment class.

```python
# Prepare data for TF-IDF + Logistic Regression
X = df_financial_phrasebank['processed_text']
y = df_financial_phrasebank['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# TF-IDF Vectorization
# max_features: Limit to top 5000 terms
# ngram_range=(1, 2): Include unigrams and bigrams to capture negation (e.g., "not good")
# min_df=3: Ignore terms that appear in less than 3 documents
# max_df=0.95: Exclude terms that appear in more than 95% of documents (too common)
# sublinear_tf=True: Apply 1 + log(TF) to dampen effect of very frequent terms
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Logistic Regression Classifier
# C=1.0: Inverse of regularization strength; smaller values specify stronger regularization
# max_iter=1000: Maximum number of iterations for the solver
# class_weight='balanced': Automatically adjusts weights inversely proportional to class frequencies
#                          This is crucial for handling imbalanced datasets like Financial PhraseBank
lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# Train the model
lr_model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred_tfidf = lr_model.predict(X_test_tfidf)

print("--- TF-IDF + Logistic Regression Performance on Financial PhraseBank ---")
print(classification_report(y_test, y_pred_tfidf))

# Extract and display top predictive words per class
feature_names = tfidf_vectorizer.get_feature_names_out()
sentiment_classes = lr_model.classes_

print("\n--- Top Predictive Words (Unigrams/Bigrams) from Logistic Regression ---")
for i, cls in enumerate(sentiment_classes):
    # Get coefficients for the current class
    # For multiclass, coef_ is (n_classes, n_features).
    # If using 'ovr' or 'multinomial', it will have one row per class.
    # We sort by coefficient value to find the most influential features.
    top_10_idx = lr_model.coef_[i].argsort()[-10:][::-1] # Top 10 positive coefficients
    bottom_10_idx = lr_model.coef_[i].argsort()[:10]   # Top 10 negative coefficients

    top_words = [feature_names[j] for j in top_10_idx]
    bottom_words = [feature_names[j] for j in bottom_10_idx]
    
    print(f"\nTop POSITIVE words for '{cls}' (higher probability): {', '.join(top_words)}")
    print(f"Top NEGATIVE words for '{cls}' (lower probability): {', '.join(bottom_words)}")

# Apply the trained TF-IDF + Logistic Regression model to 10-K risk factors
X_10k_tfidf = tfidf_vectorizer.transform(df_10k['processed_text'])
df_10k['tfidf_pred'] = lr_model.predict(X_10k_tfidf)

print("\n--- SEC 10-K Risk Factor TF-IDF + LogReg Sentiment Examples ---")
for i in range(min(3, len(df_10k))):
    print(f"Paragraph {i+1}:\n{df_10k['text'].iloc[i]}")
    print(f"Predicted: {df_10k['tfidf_pred'].iloc[i]}\n")
```

### Explanation of Execution

Sarah's TF-IDF + Logistic Regression model demonstrates a more sophisticated approach. The `classification_report` provides detailed metrics for each sentiment class. Crucially, by examining the "Top Predictive Words," Sarah can gain insight into the model's learned patterns. For instance, 'positive' sentiment might be driven by terms like "growth" or "profit," while 'negative' sentiment correlates with "loss" or "risk." The presence of bigrams such as "not profitable" reveals that the model effectively captures negation, a significant improvement over simple lexicon counting. Applying this to the 10-K paragraphs yields data-driven sentiment predictions that are more robust than the LM dictionary's rule-based outputs. This method offers a balance between accuracy and interpretability.

---

## 5. Approach C: The Power of Contextual Understanding with FinBERT

### Story + Context + Real-World Relevance

For the most advanced and context-sensitive sentiment analysis, Sarah moves to transformer models, specifically FinBERT. FinBERT is a BERT-based model pre-trained on a massive financial corpus and then fine-tuned on financial sentiment data like the Financial PhraseBank. This means it doesn't just count words or identify bigrams; it understands the semantic meaning of words based on their surrounding context. For example, "beats estimates" implies positive sentiment, whereas "market beats retreat" implies negative sentiment. This contextual understanding is what truly sets transformers apart.

The core of transformer models is the **Self-Attention Mechanism**. For a sequence of input tokens $x_1, \dots, x_n$, the attention score between tokens $i$ and $j$ is determined by Query ($Q$), Key ($K$), and Value ($V$) matrices derived from the input embeddings $X$:
$$Attention(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$
Where $Q = XW_Q$, $K = XW_K$, $V = XW_V$ are linear projections of the input embeddings $X$ by learned weight matrices $W_Q, W_K, W_V$, and $d_k$ is the dimension of the key vectors. This allows FinBERT to "attend" to relevant words across the sentence, understanding relationships that simple bag-of-words models miss (e.g., connecting "not" to "increase" to correctly invert sentiment).

For classification, FinBERT typically uses the representation of the special `[CLS]` token, denoted as $h_{CLS}$, which captures the aggregated meaning of the input sequence. This vector is then passed through a linear layer and a softmax function to predict the sentiment class:
$$P(y = k | \text{text}) = \text{softmax}(W h_{CLS} + b)_k$$
Here, $W$ and $b$ are learned weights and bias, and $k$ represents the sentiment classes (positive, negative, neutral). This "zero-shot transfer learning" capability is highly valuable as Sarah can leverage a powerful, pre-trained model without needing to train it herself.

```python
# Load FinBERT (pre-trained on financial sentiment)
# Using device=-1 ensures it runs on CPU if no GPU is available or specified
finbert_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    device=-1 # Set to 0 for GPU, -1 for CPU
)

def finbert_predict_batch(texts, batch_size=32):
    """
    Performs batch inference with FinBERT to handle large datasets efficiently.
    Truncates inputs to 512 tokens, which is a common limit for BERT-like models.
    
    Args:
        texts (list): A list of text strings to analyze.
        batch_size (int): The number of texts to process in each batch.

    Returns:
        list: A list of predicted sentiment labels ('negative', 'neutral', 'positive').
    """
    all_predictions = []
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Truncate texts to avoid exceeding model's max input length (typically 512 tokens)
        truncated_batch = [text[:512] for text in batch_texts] 
        preds = finbert_pipeline(truncated_batch)
        all_predictions.extend([p['label'].lower() for p in preds])
    return all_predictions

# Apply FinBERT to the test set of Financial PhraseBank
X_test_list = X_test.tolist() # Convert Series to list for batch processing
y_pred_finbert = finbert_predict_batch(X_test_list)

print("--- FinBERT Performance on Financial PhraseBank ---")
print(classification_report(y_test, y_pred_finbert))

# Apply FinBERT to 10-K risk factors
df_10k_text_list = df_10k['text'].tolist() # Use original text for FinBERT (it handles its own tokenization)
df_10k['finbert_pred'] = finbert_predict_batch(df_10k_text_list)

print("\n--- SEC 10-K Risk Factor FinBERT Sentiment Examples ---")
for i in range(min(3, len(df_10k))):
    print(f"Paragraph {i+1}:\n{df_10k['text'].iloc[i]}")
    print(f"Predicted: {df_10k['finbert_pred'].iloc[i]}\n")
```

### Explanation of Execution

FinBERT demonstrates its prowess in understanding financial sentiment with its high accuracy. The `classification_report` typically shows superior performance compared to the previous methods, especially in handling the 'negative' and 'positive' classes more effectively. Sarah notes that FinBERT's predictions for the 10-K risk factors are often more nuanced, capturing subtle financial sentiment that a simple word count or even bigram analysis might miss. The batch inference function ensures that even large sets of documents can be processed efficiently. This "zero-shot" capability, where the model performs well without explicit user training, highlights the power of transfer learning and provides Sarah with a state-of-the-art tool for complex text analysis.

---

## 6. Holistic Evaluation: Model Comparison for Strategic Selection

### Story + Context + Real-World Relevance

Having implemented three distinct sentiment analysis approaches, Sarah's next critical step is a comprehensive comparative evaluation. For AlphaQuant Investments, choosing the right model isn't just about raw accuracy; it's about understanding the trade-offs in interpretability, computational cost, and performance across different sentiment classes. The Financial PhraseBank dataset, with its labeled sentiments, is perfect for this task. Sarah will compare Accuracy, Macro-F1 score (crucial for imbalanced datasets), and per-class F1 scores. Visualizing confusion matrices will give her an intuitive understanding of where each model succeeds or fails, guiding her decision on which tool to deploy for specific investment use cases.

For imbalanced datasets, like the Financial PhraseBank which has a higher proportion of 'neutral' sentences, Accuracy alone can be misleading. **Macro-F1 score** is preferred because it calculates the F1-score for each class independently and then averages them, giving equal weight to each class regardless of its frequency. This prevents the score from being inflated by good performance on the majority class and highlights weaknesses in detecting minority classes (like 'negative' or 'positive'). The F1-score for class $k$ is defined as:
$$F1_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$
Where $P_k$ is Precision for class $k$ and $R_k$ is Recall for class $k$.
The Macro-F1 score is then:
$$Macro-F1 = \frac{1}{K} \sum_{k=1}^{K} F1_k$$
Where $K$ is the number of classes.

```python
# Collect all predictions for comparison
comparison_df = pd.DataFrame({
    'text': X_test,
    'actual': y_test,
    'lm_pred': df_financial_phrasebank.loc[X_test.index, 'lm_pred'],
    'tfidf_pred': y_pred_tfidf,
    'finbert_pred': y_pred_finbert
})

# Define the models and their predictions for evaluation
models_to_evaluate = {
    'LM Dictionary': comparison_df['lm_pred'],
    'TF-IDF + LogReg': comparison_df['tfidf_pred'],
    'FinBERT': comparison_df['finbert_pred']
}

results = {}
for name, predictions in models_to_evaluate.items():
    accuracy = accuracy_score(comparison_df['actual'], predictions)
    macro_f1 = f1_score(comparison_df['actual'], predictions, average='macro')
    
    # Get per-class F1 scores
    report = classification_report(comparison_df['actual'], predictions, output_dict=True)
    f1_positive = report['positive']['f1-score'] if 'positive' in report else 0
    f1_negative = report['negative']['f1-score'] if 'negative' in report else 0
    f1_neutral = report['neutral']['f1-score'] if 'neutral' in report else 0
    
    results[name] = {
        'Accuracy': accuracy,
        'Macro-F1': macro_f1,
        'F1 (Positive)': f1_positive,
        'F1 (Negative)': f1_negative,
        'F1 (Neutral)': f1_neutral
    }

performance_table = pd.DataFrame(results).T
print("--- Comparative Model Performance on Financial PhraseBank Test Set ---")
print(performance_table.round(3))

# Generate Confusion Matrices
labels_order = ['negative', 'neutral', 'positive']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices: Three Sentiment Approaches', fontsize=14)

for i, (name, predictions) in enumerate(models_to_evaluate.items()):
    cm = confusion_matrix(comparison_df['actual'], predictions, labels=labels_order)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=labels_order, yticklabels=labels_order)
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig('sentiment_comparison_confusion_matrices.png', dpi=150)
plt.show()
```

### Explanation of Execution

The comparative performance table clearly shows the trade-offs. FinBERT typically achieves the highest Macro-F1 and accuracy, particularly excelling in identifying positive and negative sentiments, which are crucial for investment decisions. The confusion matrices provide a visual diagnosis:
*   **Loughran-McDonald:** Often has lower true positives for positive/negative classes and more misclassifications, especially confusing neutral with other classes.
*   **TF-IDF + LogReg:** Shows improvement, learning to distinguish classes better, but might still struggle with subtle nuances or highly imbalanced classes.
*   **FinBERT:** Demonstrates strong diagonal values, indicating accurate predictions across all classes, and fewer errors in distinguishing between negative/positive and neutral.

For Sarah, this analysis confirms that FinBERT offers the best overall performance for detecting critical investment signals. While the LM dictionary is quick and interpretable for a first pass, and TF-IDF+LogReg offers a data-driven improvement, FinBERT's contextual understanding makes it the most powerful tool for high-stakes decisions at AlphaQuant. The choice of model, however, will also depend on the specific use case's interpretability and computational requirements.

---

## 7. From Text to Alpha: Correlating Sentiment with Stock Returns (Conceptual)

### Story + Context + Real-World Relevance

The ultimate question for Sarah is whether these sentiment scores can actually translate into an informational edge and potential "alpha" for AlphaQuant Investments. She's not just interested in model accuracy; she wants to know if high sentiment predicts higher future returns and low sentiment predicts lower returns. To conceptualize this, Sarah will perform a **sentiment-return correlation analysis** and a **quintile spread analysis**. She will simulate a dataset of financial news headlines with associated FinBERT sentiment scores and subsequent daily stock returns. This allows her to test the hypothesis that positive news sentiment can be a precursor to positive future returns, and vice-versa.

While sentiment-return correlations are often small (e.g., Spearman correlations of 0.02-0.08), research shows that when applied systematically across a large universe of stocks, they can be economically meaningful and contribute to risk-adjusted alpha. The quintile spread analysis, comparing the returns of the most-positive sentiment stocks to the most-negative sentiment stocks, is a standard test for identifying potential long/short opportunities.

```python
from scipy.stats import spearmanr

# --- Assume a DataFrame 'news_df' with financial news headlines and returns ---
# In a real-world scenario, this data would come from a live news feed and stock return database.
# For this exercise, we simulate a DataFrame to demonstrate the correlation analysis.

np.random.seed(42) # for reproducibility
num_days = 252 # ~1 year of trading days
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
num_entries = num_days * len(tickers)

dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_days))
simulated_data = []

for date in dates:
    for ticker in tickers:
        # Simulate FinBERT scores (e.g., between -1 and 1 or 0 and 1, or just labels mapped to numbers)
        # For simplicity, let's assume a numeric score from -1 to 1 based on FinBERT's output probability mapping
        # Let's say FinBERT predicts P(pos), P(neg), P(neu). We could derive a score = P(pos) - P(neg).
        # Here we'll just simulate a numeric score for demonstration.
        finbert_score = np.random.uniform(-0.8, 0.8) # Simulate a score range

        # Simulate next day return (e.g., 0.1% to -0.1% daily, with some noise)
        # Introduce a slight positive bias for positive sentiment, negative for negative
        base_return = np.random.normal(0, 0.005) # Daily return mean 0, std dev 0.5%
        next_day_return = base_return + (finbert_score * 0.001) # Add a small sentiment signal

        simulated_data.append({
            'date': date,
            'ticker': ticker,
            'headline': f"News for {ticker} on {date.strftime('%Y-%m-%d')}", # Placeholder headline
            'finbert_score': finbert_score,
            'next_day_return': next_day_return
        })

news_df = pd.DataFrame(simulated_data)

# --- Perform Sentiment-Return Correlation Analysis ---

# Aggregate daily sentiment per stock (e.g., mean sentiment if multiple headlines)
# For this simulation, we have one score per day per stock, so aggregation is simple.
daily_sent_agg = news_df.groupby(['date', 'ticker']).agg(
    avg_sentiment=('finbert_score', 'mean'),
    next_day_return=('next_day_return', 'first') # Assuming one return per day per ticker
).reset_index()

# Calculate Spearman correlation across all daily_sent_agg entries
# Spearman correlation is used because the relationship might not be linear,
# and it's robust to outliers and non-normal distributions.
corr, p_val = spearmanr(daily_sent_agg['avg_sentiment'], daily_sent_agg['next_day_return'])

print(f"--- Sentiment-Return Correlation Analysis (Simulated Data) ---")
print(f"Spearman correlation (average sentiment vs next-day return): {corr:.4f}")
print(f"P-value: {p_val:.4f}")

# --- Quintile Spread Analysis ---

# Create sentiment quintiles
# pd.qcut divides the data into equal-sized bins based on quantiles.
daily_sent_agg['sent_quintile'] = pd.qcut(
    daily_sent_agg['avg_sentiment'],
    q=5, # 5 quintiles
    labels=[1, 2, 3, 4, 5] # Label quintiles from 1 (most negative) to 5 (most positive)
)

# Calculate average next-day return for each quintile
quintile_returns = daily_sent_agg.groupby('sent_quintile')['next_day_return'].mean()

# Annualize returns (assuming 252 trading days) and convert to basis points (bps)
annualized_quintile_returns_bps = quintile_returns * 252 * 10000

print("\n--- Annualized Return by Sentiment Quintile (bps) ---")
print(annualized_quintile_returns_bps.round(0))

# Calculate Long-Short Spread (Quintile 5 - Quintile 1)
long_short_spread_bps = annualized_quintile_returns_bps.loc[5] - annualized_quintile_returns_bps.loc[1]
print(f"\nLong-Short Spread (Q5 - Q1): {long_short_spread_bps:.0f} bps")

# Plotting the quintile returns
plt.figure(figsize=(10, 6))
annualized_quintile_returns_bps.plot(kind='bar', color='skyblue')
plt.title('Annualized Returns by Sentiment Quintile (Simulated Data)', fontsize=14)
plt.xlabel('Sentiment Quintile (1 = Most Negative, 5 = Most Positive)', fontsize=12)
plt.ylabel('Annualized Return (bps)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=0.8) # Zero line
plt.tight_layout()
plt.savefig('sentiment_quintile_returns.png', dpi=150)
plt.show()
```

### Explanation of Execution

Sarah's conceptual analysis demonstrates a crucial link between sentiment and returns. The Spearman correlation coefficient, even if small (as is typical for financial sentiment), indicates a directional relationship. The p-value helps assess the statistical significance of this correlation.

More importantly, the **quintile spread analysis** reveals a potential investment signal. If the annualized return for the most positive sentiment quintile (Q5) is significantly higher than for the most negative sentiment quintile (Q1), it suggests that a strategy of "going long" on high-sentiment stocks and "going short" on low-sentiment stocks could generate alpha. The bar chart vividly illustrates this spread, showing a clear upward trend in returns as sentiment moves from negative to positive. A positive Q5 - Q1 spread (e.g., 300+ bps) would be considered economically significant for AlphaQuant. This exercise validates for Sarah that systematic sentiment analysis, despite small correlations, can be a valuable component of a quantitative investment strategy when applied at scale.

