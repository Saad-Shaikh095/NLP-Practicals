
!pip install -q indic-nlp-library nltk

from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from indicnlp.tokenize import indic_tokenize

nltk.download("punkt_tab", quiet=True)

# 1. PREPARE ENGLISH AND HINDI CORPUS

english_text = """
education helps students learn.
education helps students grow.
students learn natural language processing.
students learn language modelling.
natural language processing helps students learn.
"""

hindi_text = """
विद्यार्थी भाषा सीखते हैं।
विद्यार्थी हिंदी सीखते हैं।
विद्यार्थी भाषा समझते हैं।
हिंदी भाषा सीखना उपयोगी है।
विद्यार्थी हिंदी भाषा सीखते हैं।
"""

# 2. TOKENIZATION

english_tokens = word_tokenize(english_text.lower())

hindi_tokens = indic_tokenize.trivial_tokenize(
    hindi_text, lang="hi"
)

print("ENGLISH TOKENS:")
print(english_tokens)

print("\nHINDI TOKENS:")
print(hindi_tokens)


# 3. BUILD UNIGRAM AND BIGRAM MODELS

def build_model(tokens):

    unigram_counts = Counter(tokens)

    bigram_counts = Counter(
        (tokens[i], tokens[i + 1])
        for i in range(len(tokens) - 1)
    )

    total = len(tokens)

    unigram_prob = {
        word: count / total
        for word, count in unigram_counts.items()
    }

    previous_counts = Counter()

    for (first, second), count in bigram_counts.items():
        previous_counts[first] += count

    bigram_prob = {
        pair: count / previous_counts[pair[0]]
        for pair, count in bigram_counts.items()
    }

    return unigram_counts, bigram_counts, unigram_prob, bigram_prob


en_uni, en_bi, en_up, en_bp = build_model(english_tokens)

hi_uni, hi_bi, hi_up, hi_bp = build_model(hindi_tokens)


# 4. DISPLAY UNIGRAM PROBABILITIES

def show_unigrams(counts, probabilities, language):

    print("\n", language, "UNIGRAM PROBABILITIES")

    for word, count in counts.most_common():
        print(word, "| Count:", count,
              "| Probability:", round(probabilities[word], 4))


show_unigrams(en_uni, en_up, "ENGLISH")
show_unigrams(hi_uni, hi_up, "HINDI")


# 5. DISPLAY BIGRAM PROBABILITIES

def show_bigrams(counts, probabilities, language):

    print("\n", language, "BIGRAM PROBABILITIES")

    for pair, count in counts.most_common():
        print(pair[0], "->", pair[1],
              "| Count:", count,
              "| Probability:", round(probabilities[pair], 4))


show_bigrams(en_bi, en_bp, "ENGLISH")
show_bigrams(hi_bi, hi_bp, "HINDI")


# 6. NEXT WORD PREDICTION

def predict_next_word(previous_word, probabilities):

    candidates = []

    for (first, next_word), probability in probabilities.items():

        if first == previous_word:
            candidates.append((next_word, probability))

    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates


print("\nENGLISH NEXT WORD PREDICTION")
print("Previous word: students")
print(predict_next_word("students", en_bp))

print("\nHINDI NEXT WORD PREDICTION")
print("Previous word: विद्यार्थी")
print(predict_next_word("विद्यार्थी", hi_bp))


"""
============================================================
       PRACTICAL: ENGLISH AND HINDI N-GRAM MODEL
============================================================

1. WHAT IS N-GRAM?

An N-gram is a sequence of N consecutive words.

Types:

Unigram = One word
Example: students

Bigram = Two consecutive words
Example: students learn

Trigram = Three consecutive words
Example: students learn language

This practical uses Unigram and Bigram models.

============================================================

2. CORPUS

A corpus is a collection of text used for NLP processing.

English corpus:
Contains English sentences.

Hindi corpus:
Contains Hindi sentences.

Both corpora are used to train separate language models.

============================================================

3. TOKENIZATION

Tokenization means splitting text into smaller units
called tokens.

English:
Uses NLTK word_tokenize().

Hindi:
Uses Indic NLP trivial_tokenize().

Why use different tokenizers?

Different languages have different writing systems
and tokenization requirements.

============================================================

4. UNIGRAM MODEL

A Unigram model considers one word at a time.

It calculates the probability of each word.

FORMULA:

P(w) = Count(w) / Total number of words

Example:

If "students" occurs 5 times in a corpus
containing 100 words:

P(students) = 5 / 100 = 0.05

============================================================

5. BIGRAM MODEL

A Bigram model considers two consecutive words.

It calculates the probability of the next word
based on the previous word.

FORMULA:

P(w2 | w1) = Count(w1, w2) / Count(w1)

Example:

If:
Count(students, learn) = 4
Count(students) as first word of a bigram = 5

Then:

P(learn | students) = 4 / 5 = 0.8

============================================================

6. COUNTER()

Counter is a Python class used to count occurrences
of words or word combinations.

Example:

Counter(["cat", "dog", "cat"])

Output:
cat = 2
dog = 1

============================================================

7. UNIGRAM PROBABILITY

Unigram probability tells us how likely a particular
word is to appear in the corpus.

The model does not consider previous words.

============================================================

8. BIGRAM PROBABILITY

Bigram probability tells us how likely a word is
to appear after another particular word.

Example:

Previous word: students

Possible next words:
learn
grow
understand

The model ranks them using their probabilities.

============================================================

9. NEXT-WORD PREDICTION

Steps:

1. Take the previous word.
2. Find all bigrams starting with that word.
3. Get their probabilities.
4. Sort probabilities in descending order.
5. Return the possible next words.

============================================================

10. IMPORTANT PYTHON FUNCTIONS

Counter():
Counts word frequencies.

word_tokenize():
Tokenizes English text.

trivial_tokenize():
Tokenizes Hindi text.

lower():
Converts text into lowercase.

most_common():
Returns the most frequent items.

items():
Returns dictionary key-value pairs.

sort():
Sorts the candidate words.

lambda:
Creates a small anonymous function.

round():
Rounds decimal values.

============================================================

11. ENGLISH VS HINDI

English:
Uses the NLTK tokenizer.

Hindi:
Uses the Indic NLP tokenizer.

Both models:
- Count individual words.
- Count consecutive word pairs.
- Calculate probabilities.
- Predict possible next words.

The same mathematical formulas can be used
for both languages.

============================================================

12. LIMITATIONS

1. The corpus is very small.
2. The model considers only one previous word.
3. It cannot understand complete sentence meaning.
4. Unseen bigrams receive no probability.
5. Tokenization quality affects the result.

============================================================

13. APPLICATIONS

1. Next-word prediction.
2. Mobile keyboard suggestions.
3. Autocomplete.
4. Machine translation.
5. Speech recognition.
6. Chatbots.
7. Text generation.

============================================================

CONCLUSION:

We created separate English and Hindi language models
using Unigram and Bigram probabilities.

The models count words, calculate probabilities and
predict possible next words based on the previous word.
============================================================
"""
