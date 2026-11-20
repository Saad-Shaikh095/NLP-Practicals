
# !pip install -q pandas numpy scikit-learn

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "Data science uses python",
    "Python is a language",
    "Data science is powerful science"
]

# 1. Tokenization and Vocabulary
tokenized = [doc.lower().split() for doc in corpus]

vocab = sorted(set(word for doc in tokenized for word in doc))

n_docs = len(tokenized)

print("Vocabulary:", vocab)
print("Total Documents:", n_docs)

# 2. Term Frequency (TF)
counts = np.array([
    [doc.count(word) for word in vocab]
    for doc in tokenized
], dtype=float)

doc_lengths = np.array([len(doc) for doc in tokenized]).reshape(-1, 1)

tf = counts / doc_lengths

df_tf = pd.DataFrame(
    tf,
    columns=vocab,
    index=["Doc 1", "Doc 2", "Doc 3"]
)

print("\nTERM FREQUENCY:")
print(df_tf.round(3))

# 3. Inverse Document Frequency (IDF)
doc_freq = (counts > 0).sum(axis=0)

idf = np.log(n_docs / doc_freq)

df_idf = pd.DataFrame({
    "Document Frequency": doc_freq,
    "IDF": idf
}, index=vocab)

print("\nINVERSE DOCUMENT FREQUENCY:")
print(df_idf.round(4))

# 4. TF-IDF Calculation
tfidf = tf * idf

df_tfidf = pd.DataFrame(
    tfidf,
    columns=vocab,
    index=["Doc 1", "Doc 2", "Doc 3"]
)

print("\nTF-IDF MATRIX:")
print(df_tfidf.round(4))

# 5. IDF = 0 Case
corpus2 = [doc + " learning" for doc in corpus]

tokenized2 = [doc.lower().split() for doc in corpus2]

vocab2 = sorted(set(word for doc in tokenized2 for word in doc))

counts2 = np.array([
    [doc.count(word) for word in vocab2]
    for doc in tokenized2
], dtype=float)

tf2 = counts2 / counts2.sum(axis=1, keepdims=True)

idf2 = np.log(len(tokenized2) / (counts2 > 0).sum(axis=0))

tfidf2 = tf2 * idf2

df2 = pd.DataFrame(
    tfidf2,
    columns=vocab2
)

print("\nIDF OF LEARNING:", idf2[vocab2.index("learning")])
print(df2[["learning", "uses", "powerful"]].round(4))

# 6. Stop Word Removal
cleaned = [
    [word for word in doc if word not in ENGLISH_STOP_WORDS]
    for doc in tokenized
]

vocab3 = sorted(set(word for doc in cleaned for word in doc))

counts3 = np.array([
    [doc.count(word) for word in vocab3]
    for doc in cleaned
], dtype=float)

tf3 = counts3 / counts3.sum(axis=1, keepdims=True)

idf3 = np.log(len(cleaned) / (counts3 > 0).sum(axis=0))

tfidf3 = tf3 * idf3

df3 = pd.DataFrame(
    tfidf3,
    columns=vocab3,
    index=["Doc 1", "Doc 2", "Doc 3"]
)

print("\nTF-IDF AFTER STOP WORD REMOVAL:")
print(df3.round(3))

# 7. TF-IDF using Scikit-learn
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

X = vectorizer.fit_transform(corpus)

df_sklearn = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("\nSKLEARN TF-IDF:")
print(df_sklearn.round(4))

# 8. Raw TF-IDF without Normalization and Smoothing
raw = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    norm=None,
    smooth_idf=False
)

X_raw = raw.fit_transform(corpus)

print("\nRAW TF-IDF:")
print(pd.DataFrame(
    X_raw.toarray(),
    columns=raw.get_feature_names_out()
).round(4))

# 9. Document Ranking using Cosine Similarity
vectorizer = TfidfVectorizer(stop_words="english")

D = vectorizer.fit_transform(corpus)

query = "python language"

q = vectorizer.transform([query])

similarity = cosine_similarity(q, D).ravel()

print("\nDOCUMENT RANKING:")

for i in np.argsort(-similarity):
    print(round(similarity[i], 3), "-", corpus[i])


"""
=========================================================
EXPLANATION OF THE COMPLETE CODE
=========================================================

1. TF-IDF:

TF-IDF stands for Term Frequency-Inverse Document Frequency.

It is a text representation technique used in NLP.

It identifies how important a word is in a document
compared to a collection of documents.

A word gets:
- High weight if it is important in a document.
- Low weight if it appears in many documents.

TF-IDF has two main parts:
1. Term Frequency (TF)
2. Inverse Document Frequency (IDF)

Final formula:

TF-IDF = TF * IDF


2. TERM FREQUENCY (TF):

TF measures how many times a word appears
in a particular document.

Formula:

TF = Number of times word appears / Total words in document

Example:

Document:
Data science uses python

Total words = 4

Frequency of python = 1

TF = 1/4 = 0.25

A higher TF means the word appears more frequently
in that document.


3. INVERSE DOCUMENT FREQUENCY (IDF):

IDF measures how rare or important a word is
across all documents.

Formula:

IDF = ln(N / df)

Where:

N = Total number of documents.

df = Number of documents containing that word.

ln = Natural logarithm.

Example:

Total documents = 3

Suppose python appears in 2 documents.

IDF = ln(3/2)

IDF = 0.4055 approximately.

If a word appears in every document:

IDF = ln(3/3)
IDF = ln(1)
IDF = 0

This means the word gets zero weight.


4. TF-IDF CALCULATION:

Formula:

TF-IDF = TF * IDF

Example:

TF = 0.25
IDF = 1.0986

TF-IDF = 0.25 * 1.0986

TF-IDF = 0.2747

This gives the importance of a word
in a particular document.


5. VOCABULARY:

Vocabulary contains all unique words
from the given documents.

set() removes duplicate words.

sorted() arranges words alphabetically.

Example:

Data, Science, Data, Python

Vocabulary:
Data, Python, Science


6. DOCUMENT FREQUENCY:

Document Frequency tells us how many documents
contain a particular word.

Example:

Doc 1: Data Science
Doc 2: Data Python
Doc 3: Python Language

Document Frequency of Data = 2.

Document Frequency of Python = 2.


7. IDF = 0 CASE:

When a word appears in every document,
its IDF becomes zero.

Example:

Doc 1: Data Science Learning
Doc 2: Python Learning
Doc 3: AI Learning

IDF of Learning:

ln(3/3) = 0

Therefore:

TF-IDF = TF * 0 = 0

The word does not help distinguish documents.


8. STOP WORD REMOVAL:

Stop words are common words that may not provide
useful information for document retrieval.

Examples:
is, am, are, the, a, an, and.

ENGLISH_STOP_WORDS provides common English stop words.

Removing stop words can improve the importance
of meaningful words in some applications.


9. SCIKIT-LEARN TF-IDF:

TfidfVectorizer() converts text into TF-IDF vectors.

fit_transform() learns the vocabulary and
calculates the TF-IDF matrix.

toarray() converts the result into a normal array.

get_feature_names_out() returns the vocabulary.

Scikit-learn uses a different default formula.

Default IDF formula:

IDF = ln((1 + N) / (1 + df)) + 1

It also uses:
- Smoothed IDF.
- L2 normalization.
- Raw term counts rather than normalized TF.

Therefore, its values differ from our manual
textbook calculation.


10. COSINE SIMILARITY:

Cosine Similarity measures how similar two
numerical vectors are.

It is used to compare a search query with documents.

Its value generally ranges from 0 to 1
for non-negative TF-IDF vectors.

1 = Highly similar.
0 = No similarity.

Example:

Query = "python language"

The program compares this query with all documents
and ranks them according to similarity.

cosine_similarity() calculates similarity.

argsort() sorts the similarity values.

The highest similarity document appears first.


11. APPLICATIONS OF TF-IDF:

- Search engines
- Document ranking
- Information retrieval
- Text classification
- Keyword extraction
- Search recommendations
- News categorization


=========================================================
IMPORTANT PYTHON FUNCTIONS
=========================================================

lower()              - Converts text into lowercase.
split()              - Splits text into words.
set()                - Removes duplicate words.
sorted()             - Sorts vocabulary.
len()                - Counts documents or words.
count()              - Counts word frequency.
np.array()           - Creates a numerical array.
np.log()             - Calculates natural logarithm.
np.sum()             - Calculates sum.
TfidfVectorizer()    - Generates TF-IDF vectors.
fit_transform()      - Learns and transforms text.
toarray()            - Converts matrix into an array.
cosine_similarity()  - Calculates similarity.
np.argsort()         - Sorts array indices.

=========================================================
END OF EXPERIMENT
=========================================================
"""
