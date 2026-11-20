
# !pip install -q nltk

import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

nltk.download("brown", quiet=True)
nltk.download("punkt_tab", quiet=True)

# 1. LOAD BROWN CORPUS

corpus = brown.words()

words = [word.lower() for word in corpus]
vocab = set(words)

print("Total words:", len(words))
print("Unique words:", len(vocab))

# 2. COUNT BIGRAMS AND TRIGRAMS

bigram_counts = {}
trigram_counts = {}

for i in range(len(words) - 2):

    bigram = (words[i], words[i + 1])
    trigram = (words[i], words[i + 1], words[i + 2])

    bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1

print("\nBigram count:", bigram_counts.get(("the", "king"), 0))

# 3. PREDICT NEXT WORD

def suggest_next_word(sentence):

    tokens = word_tokenize(sentence.lower())

    if len(tokens) < 2:
        return "Please enter at least two words."

    last_two = tuple(tokens[-2:])

    suggestions = {}

    for word in vocab:

        trigram = last_two + (word,)

        trigram_count = trigram_counts.get(trigram, 0)
        bigram_count = bigram_counts.get(last_two, 0)

        if bigram_count > 0:
            probability = trigram_count / bigram_count
        else:
            probability = 0

        if probability > 0:
            suggestions[word] = probability

    top_words = sorted(
        suggestions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return top_words if top_words else "No prediction found."

# 4. TEST THE MODEL

sentences = [
    "I am the king",
    "I am the king of",
    "I am the king of france"
]

for sentence in sentences:
    print("\nInput:", sentence)
    print("Next word suggestions:", suggest_next_word(sentence))


"""
============================================================
              PRACTICAL: N-GRAM LANGUAGE MODELING
============================================================

1. WHAT IS N-GRAM?

An N-gram is a sequence of N consecutive words in a sentence.

Types:

Unigram = One word
Example: the

Bigram = Two consecutive words
Example: the king

Trigram = Three consecutive words
Example: the king of

N-gram models predict the next word using previous words.

============================================================

2. BROWN CORPUS

Brown Corpus is a collection of English text.

It is available in the NLTK library.

We use it as training data for our language model.

brown.words():
Loads words from the Brown Corpus.

lower():
Converts all words into lowercase.

set():
Removes duplicate words.

vocab:
Contains all unique words in the corpus.

============================================================

3. BIGRAM COUNT

A bigram contains two consecutive words.

Example:
The king is powerful.

Bigrams:
The king
king is
is powerful

Bigram count stores how many times each pair occurs.

Example:
("the", "king") = 10

This means the pair occurred 10 times in the corpus.

============================================================

4. TRIGRAM COUNT

A trigram contains three consecutive words.

Example:
The king is powerful.

Trigrams:
The king is
king is powerful

Trigram count stores how many times each group of
three words occurs.

============================================================

5. SLIDING WINDOW

The program moves through the corpus one word at a time.

Example:
the girl bought chocolate

Bigram:
the girl
girl bought
bought chocolate

Trigram:
the girl bought
girl bought chocolate

============================================================

6. NEXT WORD PREDICTION

The model takes the last two words of the input sentence.

It checks all vocabulary words and calculates the
probability of each word appearing next.

FORMULA:

P(w3 | w1, w2) = Count(w1, w2, w3) / Count(w1, w2)

Where:

w1 = First previous word
w2 = Second previous word
w3 = Predicted next word

Example:

P(king is powerful | king is)
= Count(king, is, powerful) / Count(king, is)

The word with the highest probability is selected.

============================================================

7. IMPORTANT PYTHON FUNCTIONS

word_tokenize():
Splits a sentence into individual tokens.

get():
Gets a dictionary value.
If the key does not exist, returns a default value.

tuple():
Creates an immutable sequence of elements.

set():
Stores unique elements.

sorted():
Sorts values.

reverse=True:
Sorts in descending order.

lambda:
Creates a small anonymous function.

enumerate():
Returns index and value together.

============================================================

8. WHAT IS A LANGUAGE MODEL?

A language model learns patterns in language and
calculates the probability of word sequences.

It can be used for:
- Next-word prediction.
- Autocomplete.
- Text generation.
- Speech recognition.
- Machine translation.
- Chatbots.

============================================================

9. LIMITATIONS

1. It depends on the training corpus.
2. It cannot easily predict unseen word combinations.
3. It considers only the previous two words.
4. It does not fully understand sentence meaning.
5. It has no smoothing, so unseen trigrams get zero probability.

============================================================

10. CONCLUSION

The N-gram language model uses the Brown Corpus to
calculate bigram and trigram frequencies.

It predicts the next word by calculating conditional
probability using the previous two words.

============================================================
"""
