
# !pip install -q nltk spacy wordcloud
# !python -m spacy download en_core_web_sm -q

import nltk
import pandas as pd
import spacy
import matplotlib.pyplot as plt

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from wordcloud import WordCloud

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

nlp = spacy.load("en_core_web_sm")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

text = "Don't blink! The data scientists are buying new computers for their user studies."

# 1. Tokenization
print("\nTOKENIZATION")
print("Split:", text.split())
print("Word Tokenization:", word_tokenize(text))
print("Sentence Tokenization:", sent_tokenize(text))

tokens = word_tokenize(text)

# 2. Stemming and Lemmatization
print("\nSTEMMING AND LEMMATIZATION")

data = []

for word in tokens:
    if word.isalnum():
        data.append([
            word,
            stemmer.stem(word),
            lemmatizer.lemmatize(word.lower())
        ])

print(pd.DataFrame(data, columns=["Word", "Stem", "Lemma"]))

# 3. POS-aware Lemmatization
print("\nPOS AWARE LEMMATIZATION")

def get_pos(tag):
    return {"J": wn.ADJ, "V": wn.VERB, "R": wn.ADV}.get(tag[0], wn.NOUN)

data = []

for word, tag in pos_tag(tokens):
    if word.isalnum():
        data.append([
            word,
            tag,
            stemmer.stem(word),
            lemmatizer.lemmatize(word.lower()),
            lemmatizer.lemmatize(word.lower(), get_pos(tag))
        ])

print(pd.DataFrame(data, columns=[
    "Word", "POS", "Stem", "Normal Lemma", "POS Lemma"
]))

# 4. Stop Word Removal
print("\nSTOP WORD REMOVAL")

stop_words = set(stopwords.words("english"))

filtered = [
    word for word in tokens
    if word.lower() not in stop_words and word.isalnum()
]

print("Original:", tokens)
print("Filtered:", filtered)

# 5. Morphological Analysis using spaCy
print("\nMORPHOLOGICAL ANALYSIS")

doc = nlp(text)

data = []

for token in doc:
    if not token.is_space:
        data.append([
            token.text,
            token.lemma_,
            token.pos_,
            token.is_stop,
            str(token.morph)
        ])

print(pd.DataFrame(data, columns=[
    "Word", "Lemma", "POS", "Stopword", "Morphology"
]))

# 6. Inflection and Derivation
print("\nINFLECTION AND DERIVATION")

words = [
    "studies", "studying", "studied",
    "happiness", "unhappy", "computer", "computation"
]

data = []

for word in words:
    data.append([
        word,
        stemmer.stem(word),
        nlp(word)[0].lemma_
    ])

print(pd.DataFrame(data, columns=["Word", "Stem", "Lemma"]))

# 7. NLTK vs spaCy
print("\nNLTK VS SPACY")

nltk_tokens = word_tokenize(text)
spacy_tokens = [token.text for token in nlp(text)]

print("NLTK:", nltk_tokens)
print("spaCy:", spacy_tokens)
print("Same Tokens:", nltk_tokens == spacy_tokens)

test = "The U.S. isn't a state-of-the-art lab."

print("NLTK:", word_tokenize(test))
print("spaCy:", [token.text for token in nlp(test)])

# 8. Word Cloud
print("\nWORD CLOUD")

paragraph = """
Data scientists study data. The scientists collect data,
clean data and build models. Models learn from data,
and good models help scientists make decisions.
Studying user behaviour helps scientists improve the models.
"""

doc = nlp(paragraph)

lemmas = [
    token.lemma_.lower()
    for token in doc
    if token.is_alpha and not token.is_stop
]

print("Processed Words:", lemmas)

wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(" ".join(lemmas))

plt.figure(figsize=(10, 5))
plt.imshow(wc)
plt.axis("off")
plt.show()

"""
=========================================================
EXPLANATION OF THE COMPLETE CODE
=========================================================

1. NLP:
NLP stands for Natural Language Processing.
It is a branch of AI that helps computers understand
and process human languages.

Examples:
- Google Translate
- ChatGPT
- Voice Assistants

2. TOKENIZATION:
Tokenization means breaking a sentence into smaller
parts called tokens.

Types:
- Word Tokenization: Breaks sentences into words.
- Sentence Tokenization: Breaks paragraphs into sentences.

split() separates text using spaces.
word_tokenize() separates words and punctuation.

3. STEMMING:
Stemming removes prefixes or suffixes from words
to get their root-like form.

Examples:
Playing -> play
Studies -> studi
Computers -> comput

PorterStemmer() is used for stemming.

Stemming is fast but sometimes produces incomplete words.

4. LEMMATIZATION:
Lemmatization converts a word into its meaningful
dictionary base form.

Examples:
Studies -> study
Running -> run (when identified as a verb)
Better -> good (when identified as an adjective)

WordNetLemmatizer() performs lemmatization.

It can use POS information to find the correct base form.

5. POS TAGGING:
POS stands for Part of Speech.

It identifies the grammatical category of a word.

Examples:
Noun: Boy
Verb: Running
Adjective: Beautiful
Adverb: Quickly

pos_tag() is used to identify POS tags.

6. STOP WORD REMOVAL:
Stop words are common words that may be removed
during text preprocessing.

Examples:
is, am, are, the, a, an, and, in.

stopwords.words("english") provides English stop words.

Stop-word removal reduces unnecessary words.

Important:
Removing words like "not" can change the meaning
of a sentence.

7. MORPHOLOGICAL ANALYSIS:
Morphological analysis studies the internal structure
and grammatical properties of words.

It identifies:
- Tense
- Number
- Verb form
- Aspect
- Other grammatical features

spaCy provides morphological information.

Example:
Scientists -> Number = Plural
Buying -> Verb form = Participle

8. INFLECTION:
Inflection changes the grammatical form of a word
without changing its basic word class.

Examples:
Study -> Studies
Play -> Playing
Walk -> Walked

9. DERIVATION:
Derivation creates a new word from an existing word.
It can change the meaning or word class.

Examples:
Happy -> Happiness
Teach -> Teacher
Compute -> Computer

10. NLTK:
NLTK stands for Natural Language Toolkit.

It is a Python library used for NLP tasks such as:
- Tokenization
- Stemming
- Lemmatization
- POS Tagging

11. SPACY:
spaCy is a Python library used for NLP processing.

It provides:
- Tokenization
- POS Tagging
- Lemmatization
- Morphological Analysis

12. NLTK VS SPACY:
Both are NLP libraries.

NLTK is commonly used for learning and experimenting.
spaCy is commonly used for practical NLP applications.

13. WORD CLOUD:
A Word Cloud is a visual representation of words.

Frequently occurring words generally appear larger.

WordCloud() creates the word cloud.
matplotlib displays the generated image.

=========================================================
IMPORTANT PYTHON FUNCTIONS
=========================================================

word_tokenize()  - Splits text into words.
sent_tokenize()  - Splits text into sentences.
stem()           - Performs stemming.
lemmatize()      - Performs lemmatization.
pos_tag()        - Identifies parts of speech.
stopwords.words()- Returns stop words.
spacy.load()     - Loads an NLP model.
nlp()            - Processes text using spaCy.
isalpha()        - Checks alphabetic characters.
isalnum()        - Checks letters and numbers.
WordCloud()      - Generates a word cloud.

=========================================================
END OF EXPERIMENT
=========================================================

"""