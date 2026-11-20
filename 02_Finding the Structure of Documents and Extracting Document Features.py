
import nltk

nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize

document = """
Natural Language Processing (NLP) is a subfield of artificial intelligence. It focuses on the interaction between computers and humans.

Dr. Smith from Acme Corp announced a new breakthrough on October 22, 2026. The research facility is located in San Francisco, California.
"""

# 1. Paragraph Segmentation
paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]

print("Total Paragraphs:", len(paragraphs))

for i, paragraph in enumerate(paragraphs, 1):

    print("\nParagraph", i, ":", paragraph)

    # 2. Sentence Tokenization
    sentences = sent_tokenize(paragraph)

    print("Total Sentences:", len(sentences))

    for j, sentence in enumerate(sentences, 1):

        print("\nSentence", j, ":", sentence)

        # 3. Word Tokenization
        words = word_tokenize(sentence)

        print("Words:", words)

        # 4. POS Tagging
        tags = nltk.pos_tag(words)

        print("POS Tags:", tags)

        # 5. Display First Five Features
        print("First Five Features:", tags[:5])

"""
=========================================================
EXPLANATION OF THE COMPLETE CODE
=========================================================

1. DOCUMENT:
A document is a collection of text containing paragraphs,
sentences and words.

In this practical, we use a sample document containing
information about Natural Language Processing.

2. PARAGRAPH SEGMENTATION:
Paragraph segmentation means dividing a document
into separate paragraphs.

We use:
document.split("\n\n")

It separates paragraphs using double newlines.

strip() removes unnecessary spaces.
len() counts the total number of paragraphs.

3. SENTENCE TOKENIZATION:
Sentence tokenization means breaking a paragraph
into individual sentences.

We use:
sent_tokenize(paragraph)

Example:
Input:
I like Python. I am learning NLP.

Output:
['I like Python.', 'I am learning NLP.']

It also handles abbreviations like Dr. better
than simple full-stop splitting.

4. WORD TOKENIZATION:
Word tokenization means breaking a sentence
into individual words and punctuation marks.

We use:
word_tokenize(sentence)

Example:
Input:
I love NLP.

Output:
['I', 'love', 'NLP', '.']

5. POS TAGGING:
POS stands for Part of Speech.

It identifies the grammatical category of each word.

We use:
nltk.pos_tag(words)

Examples:
NN   = Noun
NNS  = Plural Noun
VB   = Verb
VBD  = Past Tense Verb
JJ   = Adjective
RB   = Adverb
NNP  = Proper Noun
DT   = Determiner

Example:
Python is powerful.

Python -> NNP
is     -> VBZ
powerful -> JJ

6. FEATURE EXTRACTION:
Feature extraction means identifying useful
information from a document.

In this practical, we extract:
- Paragraph count
- Sentence count
- Individual words
- POS tags

These features help computers understand
the structure and grammatical information
of a document.

7. enumerate():
enumerate() provides both the index and value
while looping through a list.

Example:
enumerate(["Apple", "Mango"])

It gives:
1. Apple
2. Mango

In our code, it helps number paragraphs and sentences.

8. tags[:5]:
This displays only the first five POS-tagged words
from the sentence.

9. NLTK:
NLTK stands for Natural Language Toolkit.

It is a Python library used for Natural Language
Processing tasks such as:
- Tokenization
- POS Tagging
- Stemming
- Lemmatization

=========================================================
IMPORTANT PYTHON FUNCTIONS
=========================================================

split()          - Divides text into parts.
strip()          - Removes extra spaces.
len()            - Counts elements.
sent_tokenize()  - Splits text into sentences.
word_tokenize()  - Splits sentences into words.
pos_tag()        - Identifies grammatical categories.
enumerate()      - Gives index and value.
tags[:5]         - Displays first five elements.

=========================================================
END OF EXPERIMENT
=========================================================
"""