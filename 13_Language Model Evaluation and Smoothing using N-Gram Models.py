
import math
from collections import Counter

# 1. TRAINING AND TESTING DATA

train_text = "i love nlp and i love machine learning"
test_text = "i love learning"

train_tokens = train_text.split()
test_tokens = test_text.split()

print("Training Corpus:", train_text)
print("Training Tokens:", train_tokens)
print("Testing Sentence:", test_text)


# 2. GENERATE N-GRAMS

def generate_ngrams(tokens, n):

    return [
        tuple(tokens[i:i+n])
        for i in range(len(tokens)-n+1)
    ]


# 3. TRAIN N-GRAM MODEL

def train_model(tokens, n):

    ngram_counts = Counter(generate_ngrams(tokens, n))

    if n > 1:
        context_counts = Counter(
            generate_ngrams(tokens, n-1)
        )
    else:
        context_counts = None

    vocabulary = set(tokens)

    return ngram_counts, context_counts, vocabulary


# 4. DISPLAY N-GRAM COUNTS

for n in [1, 2, 3]:

    ng, ctx, vocab = train_model(train_tokens, n)

    print("\n", n, "-GRAM COUNTS")

    for gram, count in ng.items():
        print(gram, ":", count)


# 5. ADD-ONE SMOOTHING

def probability(ngram, ng_counts, ctx_counts, vocab):

    V = len(vocab)

    # Unigram probability
    if len(ngram) == 1:

        total = sum(ng_counts.values())

        return (ng_counts[ngram] + 1) / (total + V)

    # Bigram and Trigram probability
    context = ngram[:-1]

    return (
        ng_counts[ngram] + 1
    ) / (
        ctx_counts[context] + V
    )


# 6. CALCULATE PERPLEXITY

def calculate_perplexity(test_tokens, n, ng_counts, ctx_counts, vocab):

    test_ngrams = generate_ngrams(test_tokens, n)

    if not test_ngrams:
        return float("inf")

    log_sum = 0

    for gram in test_ngrams:

        p = probability(
            gram, ng_counts, ctx_counts, vocab
        )

        log_sum += math.log(p)

    return math.exp(-log_sum / len(test_ngrams))


# 7. EVALUATE ALL THREE MODELS

for n in [1, 2, 3]:

    ng, ctx, vocab = train_model(train_tokens, n)

    ppl = calculate_perplexity(
        test_tokens, n, ng, ctx, vocab
    )

    print(
        "\n", n, "-gram perplexity:",
        round(ppl, 4)
    )


"""
============================================================
       PRACTICAL 13: N-GRAM EVALUATION AND SMOOTHING
============================================================

1. WHAT IS A LANGUAGE MODEL?

A language model predicts the probability of
a word or sequence of words.

Example:
I love NLP.

The model learns word patterns from training text.

============================================================

2. WHAT IS TRAINING DATA?

Training data is the text used to train the model.

Here:
i love nlp and i love machine learning

============================================================

3. WHAT IS TESTING DATA?

Testing data is separate text used to evaluate
how well the trained model predicts words.

Here:
i love learning

============================================================

4. WHAT IS N-GRAM?

An N-gram is a sequence of N consecutive words.

Unigram = One word
Bigram = Two consecutive words
Trigram = Three consecutive words

Example:
i love nlp

Unigrams:
i, love, nlp

Bigrams:
i love
love nlp

Trigrams:
i love nlp

============================================================

5. WHAT IS N-GRAM MODEL TRAINING?

It means counting how frequently each N-gram
and its preceding context appear in the corpus.

Counter() is used to count frequencies.

============================================================

6. WHAT IS ADD-ONE SMOOTHING?

Add-one smoothing is also called Laplace smoothing.

It adds 1 to every N-gram count to avoid zero
probabilities for unseen word combinations.

FORMULA:

Bigram:
P(w2|w1) = (C(w1,w2) + 1) / (C(w1) + V)

V = Vocabulary size

For Trigram:
P(w3|w1,w2) =
(C(w1,w2,w3) + 1) / (C(w1,w2) + V)

============================================================

7. WHAT IS PERPLEXITY?

Perplexity measures how well a language model
predicts a test sentence.

Lower perplexity generally means better prediction.

FORMULA:

PP = exp(-1/N * SUM(log(P(w))))

N = Number of test N-grams
P(w) = Probability of each test N-gram

============================================================

8. WHAT DOES LOW PERPLEXITY MEAN?

Low perplexity:
The model predicts the test text more confidently.

High perplexity:
The model is more uncertain about the test text.

============================================================

9. WHY DO WE USE LOGARITHM?

Multiplying many small probabilities can produce
very small numbers.

Logarithms convert multiplication into addition
and make calculation more stable.

============================================================

10. WHAT IS VOCABULARY?

Vocabulary is the collection of unique words
in the training corpus.

set() removes duplicate words.

============================================================

11. WHAT IS COUNTER?

Counter is a Python collection used to count
how many times each word or N-gram occurs.

============================================================

12. IMPORTANT PYTHON FUNCTIONS

split():
Splits text into words.

tuple():
Creates an ordered group of words.

Counter():
Counts frequencies.

set():
Stores unique words.

math.log():
Calculates natural logarithm.

math.exp():
Calculates exponential value.

len():
Returns the number of elements.

range():
Generates a sequence of numbers.

============================================================

13. APPLICATIONS

1. Speech recognition.
2. Machine translation.
3. Text prediction.
4. Autocomplete.
5. Chatbots.
6. Spelling correction.

============================================================

14. LIMITATIONS

1. Small training data gives unreliable results.
2. N-gram models cannot understand full meaning.
3. Add-one smoothing can assign too much
   probability to unseen N-grams.
4. Higher N-grams need more training data.

============================================================

CONCLUSION:

We trained Unigram, Bigram and Trigram models
using a training corpus.

We applied Add-one smoothing to avoid zero
probabilities and calculated perplexity to
evaluate the models on testing data.

============================================================
"""
