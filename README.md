[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mW4WPbr-)
# A06 - Deep Feelings

Aidan Fernandez - Undergraduate

CS 362/562 — Spring 2026 Sentiment Classification Assignment.

## Implementation Overview

All three classification systems are implemented in [`deep_feelings.py`](deep_feelings.py) using **Logistic Regression** as the classifier across all systems for fair comparison.

### How to Run

```bash
python deep_feelings.py
# or with explicit file paths:
python deep_feelings.py A06_train.csv A06_test.csv
```

Requirements: `scikit-learn`, `nltk`, `spacy`, `pandas`, `numpy`  
SpaCy model: `en_core_web_md` (`python -m spacy download en_core_web_md`)

---

## Systems

### 1. Baseline System (Bag-of-Words)
- **Features:** CountVectorizer with unigrams (`min_df=2`)
- **Preprocessing:** Lowercased; URLs, @mentions, and non-alphabetic characters removed
- **Accuracy: 64.71%**

### 2. Enhanced System
Three enhancements were applied over the baseline:
1. **Stopword removal** — filtered NLTK English stopwords to reduce noise from high-frequency words (e.g., "the", "is") that carry little sentiment signal
2. **Lemmatization** — applied WordNet lemmatizer to collapse inflected forms (e.g., "running" → "run") and reduce vocabulary sparsity
3. **Bigram features** — extended `ngram_range` to `(1,2)` to capture two-word sentiment phrases (e.g., "not good", "very happy")

- **Accuracy: 64.38%**

### 3. Embeddings System
- **Features:** Mean of spaCy `en_core_web_md` word vectors (300-dimensional dense vectors) for each document
- **Accuracy: 58.35%**

---

## Results Summary

| System | Accuracy | vs. Baseline |
|--------|----------|--------------|
| Baseline (BoW unigrams) | 64.71% | — |
| Enhanced (stopwords + lemmatization + bigrams) | 64.38% | ↓ 0.33pp |
| Embeddings (spaCy en_core_web_md) | 58.35% | ↓ 6.35pp |

---

## Analysis

### Enhanced System — did NOT lead to increased performance (−0.33pp)

**Why did the changes not result in increased performance?**

The stopword removal and lemmatization likely hurt slightly in this domain because tweets and short social-media text are already terse — every word tends to carry signal. Removing stopwords like "not" or "no" (which sometimes slip through NLTK's list depending on casing/tokenization) can inadvertently delete negation cues critical for sentiment. For example, "not bad" losing "not" becomes just "bad," flipping the polarity. Lemmatization can also conflate words that have subtly different sentiment connotations.

The bigram extension should theoretically help by capturing negation ("not good") but also drastically expands the feature space. With `min_df=2`, many meaningful bigrams are still pruned, while common neutral bigrams inflate the dimensionality and may introduce noise that counteracts the benefits.

**How can the performance be improved?**

- Use a smarter negation-handling strategy: instead of stripping stopwords blindly, prepend "NOT_" to tokens following negation words (e.g., "not good" → "NOT_good").
- Switch to TF-IDF weighting instead of raw counts to down-weight very frequent terms while keeping stopwords with sentiment value.
- Tune the `min_df` threshold for bigrams separately from unigrams.
- Add character n-grams to handle informal spelling (e.g., "greaaaat", "loool").

---

### Embeddings System — did NOT lead to increased performance (−6.35pp)

**Why did the changes not result in increased performance?**

The mean-vector approach compresses a full document into a single fixed-size vector by averaging all token embeddings. For short tweets, this averaging can destroy the positional and compositional information that determines sentiment. A positive word and a negative word in the same tweet average out to something neutral, losing the distinction between "not bad" and "really good." The `en_core_web_md` vectors are trained on general web text and may not represent Twitter-specific vocabulary well — slang, hashtags, and abbreviated forms often lack meaningful embeddings and get mapped to zero vectors, degrading the document representation.

In contrast, the bag-of-words model directly counts occurrences of sentiment-bearing words and is not harmed by vocabulary mismatch.

**How can the performance be improved?**

- Use a transformer-based model (e.g., `bert-base-uncased` or a Twitter-tuned model like `bertweet-base`) instead of averaged static vectors — contextual embeddings handle negation and word order naturally.
- Fine-tune embeddings on the training data rather than using frozen pre-trained vectors.
- Use a tweet-specific spaCy/gensim model trained on Twitter corpora.
- Concatenate embedding features with BoW features rather than replacing BoW entirely, to combine the strengths of both representations.

---

## LLM Usage

**Which LLM(s) did you use?**  
Claude Sonnet 4.6 (`claude-sonnet-4-6`) via Claude Code CLI.

**What was your series of prompts?**  
1. Provided the assignment PDF and asked to complete the assignment as an undergraduate student.
2. The model read the data files, checked available packages, installed missing ones, then generated the full implementation and README.

**Did you find this use helpful?**  
Yes. It significantly sped up the boilerplate (data loading, pipeline setup, evaluation printing) and helped structure the analysis section.

**What did you learn from its use?**  
LLMs are effective at generating standard scikit-learn pipeline code and suggesting reasonable preprocessing choices, but the analysis reasoning still requires understanding of NLP fundamentals (e.g., why averaging embeddings loses compositional meaning). The results were validated by running the code and observing actual accuracy numbers, which is necessary since LLM-generated code can contain subtle bugs.

> **Note:** The implementation in `deep_feelings.py` was generated with LLM assistance (Claude Sonnet 4.6). This is indicated in the file-level docstring.
