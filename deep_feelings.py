"""
CS 362/562 - Assignment 6: Deep Feelings
Sentiment classification using bag-of-words, enhanced features, and word embeddings.

LLM Usage: Claude Sonnet 4.6 (claude-sonnet-4-6) was used to assist with
implementation structure and make the code more Pythonic.
"""

import sys
import re
import csv
import ssl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# SSL workaround for macOS NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath):
    """Load CSV with columns: sentiment (-1/0/1), text."""
    labels, texts = [], []
    with open(filepath, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                label = int(row[0])
            except ValueError:
                continue
            text = ','.join(row[1:]).strip()
            labels.append(label)
            texts.append(text)
    return texts, labels


# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')


def clean(text):
    """Remove URLs, @mentions, and non-alphabetic characters."""
    text = URL_RE.sub('', text)
    text = MENTION_RE.sub('', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()


def preprocess_baseline(text):
    return clean(text)


def preprocess_enhanced(text):
    """Remove stopwords and lemmatize after cleaning."""
    tokens = clean(text).split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# System 1: Baseline (bag-of-words, unigrams)
# ---------------------------------------------------------------------------

def build_baseline(train_texts, train_labels):
    pipeline = Pipeline([
        ('vec', CountVectorizer(preprocessor=preprocess_baseline, min_df=2)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(train_texts, train_labels)
    return pipeline


# ---------------------------------------------------------------------------
# System 2: Enhanced (stopword removal + lemmatization + bigrams)
# ---------------------------------------------------------------------------

def build_enhanced(train_texts, train_labels):
    """
    Enhancements over baseline:
      1. Stopword removal   - filters noise words that carry little sentiment signal
      2. Lemmatization      - collapses inflected forms (running -> run) to reduce sparsity
      3. Bigram features    - captures two-word phrases (e.g. "not good") missed by unigrams
    """
    pipeline = Pipeline([
        ('vec', CountVectorizer(
            preprocessor=preprocess_enhanced,
            ngram_range=(1, 2),  # unigrams + bigrams
            min_df=2,
        )),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(train_texts, train_labels)
    return pipeline


# ---------------------------------------------------------------------------
# System 3: Embeddings (spaCy sentence vectors)
# ---------------------------------------------------------------------------

def get_spacy_vectors(texts, nlp):
    """Return mean word-vector for each text (300-dim for en_core_web_md)."""
    vectors = []
    for doc in nlp.pipe(texts, batch_size=256, disable=['parser', 'ner']):
        vectors.append(doc.vector)
    return np.array(vectors)


def build_embeddings(train_texts, train_labels, nlp):
    """
    Enhancement: replace sparse bag-of-words with dense pre-trained word vectors.
    Each document is represented as the mean of its token vectors (spaCy en_core_web_md).
    This captures semantic similarity not visible in surface word-form counts.
    """
    X_train = get_spacy_vectors(train_texts, nlp)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)
    return clf


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(name, model, test_texts, test_labels, nlp=None):
    if nlp is not None:
        X_test = get_spacy_vectors(test_texts, nlp)
        preds = model.predict(X_test)
    else:
        preds = model.predict(test_texts)
    acc = accuracy_score(test_labels, preds)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print()
    print(classification_report(test_labels, preds,
                                target_names=['Negative', 'Neutral', 'Positive'],
                                zero_division=0))
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_path = 'A06_train.csv'
    test_path  = 'A06_test.csv'

    if len(sys.argv) == 3:
        train_path, test_path = sys.argv[1], sys.argv[2]
    elif len(sys.argv) != 1:
        print(f"Usage: python {sys.argv[0]} [train.csv test.csv]")
        sys.exit(1)

    print("Loading data...")
    train_texts, train_labels = load_data(train_path)
    test_texts,  test_labels  = load_data(test_path)
    print(f"  Train: {len(train_texts)} samples | Test: {len(test_texts)} samples")

    # --- Baseline ---
    print("\nTraining baseline system (bag-of-words + LogisticRegression)...")
    baseline = build_baseline(train_texts, train_labels)
    acc_baseline = evaluate("Baseline (BoW unigrams)", baseline, test_texts, test_labels)

    # --- Enhanced ---
    print("Training enhanced system (stopwords + lemmatization + bigrams)...")
    enhanced = build_enhanced(train_texts, train_labels)
    acc_enhanced = evaluate("Enhanced (stopwords + lemmatization + bigrams)",
                            enhanced, test_texts, test_labels)

    # --- Embeddings ---
    print("Loading spaCy model (en_core_web_md)...")
    nlp = spacy.load('en_core_web_md')
    print("Training embeddings system (spaCy mean vectors)...")
    emb_model = build_embeddings(train_texts, train_labels, nlp)
    acc_emb = evaluate("Embeddings (spaCy en_core_web_md)", emb_model,
                       test_texts, test_labels, nlp=nlp)

    # --- Summary ---
    print("\n" + "="*55)
    print("  Summary")
    print("="*55)
    print(f"  Baseline  accuracy: {acc_baseline*100:.2f}%")
    print(f"  Enhanced  accuracy: {acc_enhanced*100:.2f}%  "
          f"({'↑' if acc_enhanced > acc_baseline else '↓'}"
          f" {abs(acc_enhanced - acc_baseline)*100:.2f}pp vs baseline)")
    print(f"  Embedding accuracy: {acc_emb*100:.2f}%  "
          f"({'↑' if acc_emb > acc_baseline else '↓'}"
          f" {abs(acc_emb - acc_baseline)*100:.2f}pp vs baseline)")
    print()


if __name__ == '__main__':
    main()
