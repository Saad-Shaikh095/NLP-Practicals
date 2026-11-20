
import nltk
from nltk.corpus import brown, reuters
from collections import Counter

# Download datasets
nltk.download("brown", quiet=True)
nltk.download("reuters", quiet=True)

# 1. CLEAN THE SENTENCES

def clean_sentences(corpus):

    sentences = []

    for sentence in corpus:

        words = [
            word.lower()
            for word in sentence
            if word.isalpha()
        ]

        if len(words) > 1:
            sentences.append(words)

    return sentences


# 2. LOAD CORPORA

general_sentences = clean_sentences(brown.sents())
domain_sentences = clean_sentences(reuters.sents())

print("Brown sentences:", len(general_sentences))
print("Reuters sentences:", len(domain_sentences))


# 3. CREATE COMMON VOCABULARY

vocab = set()

for sentence in general_sentences + domain_sentences:
    vocab.update(sentence)

print("Vocabulary size:", len(vocab))


# 4. GENERATE N-GRAMS

def generate_ngrams(tokens, n):

    return [
        tuple(tokens[i:i+n])
        for i in range(len(tokens)-n+1)
    ]


# 5. TRAIN N-GRAM MODEL

def train_model(sentences, n):

    ngram_counts = Counter()
    context_counts = Counter()

    for sentence in sentences:

        ngram_counts.update(generate_ngrams(sentence, n))

        if n > 1:
            context_counts.update(
                generate_ngrams(sentence, n-1)
            )

    return ngram_counts, context_counts


# 6. CALCULATE ADD-ONE SMOOTHING PROBABILITY

def calculate_probability(ngram, ng_counts, ctx_counts, vocab):

    V = len(vocab)

    if len(ngram) == 1:

        total = sum(ng_counts.values())

        return (ng_counts[ngram] + 1) / (total + V)

    context = ngram[:-1]

    return (
        ng_counts[ngram] + 1
    ) / (
        ctx_counts[context] + V
    )


# 7. TRAIN GENERAL AND DOMAIN MODELS

n = 2

general_ng, general_ctx = train_model(general_sentences, n)

domain_ng, domain_ctx = train_model(domain_sentences, n)


# 8. SELECT CONTEXT AND WORD

context = ("the",)
word = "market"

gram = context + (word,)


# 9. CALCULATE GENERAL AND DOMAIN PROBABILITIES

P_general = calculate_probability(
    gram, general_ng, general_ctx, vocab
)

P_domain = calculate_probability(
    gram, domain_ng, domain_ctx, vocab
)


# 10. INTERPOLATION

lambda_value = 0.7

P_adapted = (
    lambda_value * P_domain
    + (1 - lambda_value) * P_general
)


# 11. DISPLAY RESULTS

print("\nN-GRAM MODEL: BIGRAM")
print("Selected N-gram:", gram)

print("\nGENERAL MODEL")
print("Count:", general_ng[gram])
print("Context Count:", general_ctx[context])
print("Probability:", round(P_general, 6))

print("\nDOMAIN MODEL")
print("Count:", domain_ng[gram])
print("Context Count:", domain_ctx[context])
print("Probability:", round(P_domain, 6))

print("\nADAPTED MODEL")
print("Lambda:", lambda_value)
print("Adapted Probability:", round(P_adapted, 6))


# 12. COMPARE DIFFERENT WORDS

words = ["company", "government", "bank"]

print("\nCOMPARISON OF ADAPTED PROBABILITIES")

for word in words:

    gram = ("the", word)

    P_general = calculate_probability(
        gram, general_ng, general_ctx, vocab
    )

    P_domain = calculate_probability(
        gram, domain_ng, domain_ctx, vocab
    )

    P_adapted = (
        lambda_value * P_domain
        + (1 - lambda_value) * P_general
    )

    print("\nWord:", word)
    print("General:", round(P_general, 6))
    print("Domain:", round(P_domain, 6))
    print("Adapted:", round(P_adapted, 6))


"""
============================================================
           NLP: LANGUAGE MODEL ADAPTATION
============================================================

1. WHAT IS LANGUAGE MODEL ADAPTATION?

Language Model Adaptation means modifying a general
language model so that it works better for a particular
domain or subject.

Example:

General language:
Everyday English sentences.

Domain-specific language:
News, business, medical or sports sentences.

============================================================

2. WHAT IS BROWN CORPUS?

Brown Corpus is a collection of general English text.

It is used to train a general language model.

============================================================

3. WHAT IS REUTERS CORPUS?

Reuters Corpus contains news-related text.

It is useful for training a domain-specific model.

============================================================

4. WHAT IS A CORPUS?

A corpus is a collection of text or sentences
used for NLP processing.

============================================================

5. WHAT IS VOCABULARY?

Vocabulary is the collection of unique words
present in a corpus.

set() is used to remove duplicate words.

============================================================

6. WHAT IS N-GRAM?

An N-gram is a sequence of N consecutive words.

Unigram = One word
Bigram = Two words
Trigram = Three words

Example:

Sentence: The market is growing.

Unigrams:
The, market, is, growing

Bigrams:
The market
market is
is growing

Trigrams:
The market is
market is growing

============================================================

7. WHAT IS N-GRAM MODEL TRAINING?

It means counting how frequently N-grams
and their preceding contexts occur in the corpus.

Counter() is used to count frequencies.

============================================================

8. WHAT IS CONTEXT?

Context means the previous word or words
used to predict the next word.

For Bigram:
Context = One previous word.

Example:
The market

Context = The
Next word = Market

============================================================

9. WHAT IS PROBABILITY?

Probability tells us how likely a word or
word sequence is to occur.

Bigram formula:

P(w2 | w1) = Count(w1,w2) / Count(w1)

============================================================

10. WHAT IS ADD-ONE SMOOTHING?

Add-one smoothing is a technique used to avoid
zero probability for unseen N-grams.

Formula:

P(w2 | w1) =
(Count(w1,w2) + 1) /
(Count(w1) + V)

V = Vocabulary size

We add 1 to the numerator and vocabulary size
to the denominator.

============================================================

11. WHAT IS GENERAL LANGUAGE MODEL?

A general language model is trained on a broad
collection of text.

Here, Brown Corpus is used.

============================================================

12. WHAT IS DOMAIN-SPECIFIC MODEL?

A domain-specific model is trained on text
from a particular field.

Here, Reuters Corpus is used.

============================================================

13. WHAT IS INTERPOLATION?

Interpolation combines probabilities from
two or more language models.

Formula:

P(adapted) =
lambda * P(domain)
+ (1-lambda) * P(general)

============================================================

14. WHAT IS LAMBDA?

Lambda controls the importance of each model.

If lambda = 0.7:

70% importance to domain model.
30% importance to general model.

If lambda = 0:
Only general model.

If lambda = 1:
Only domain model.

============================================================

15. WHY IS LANGUAGE MODEL ADAPTATION USED?

1. To improve domain-specific predictions.
2. To understand technical vocabulary.
3. To improve text prediction.
4. To adapt general models to specialized fields.

============================================================

16. PYTHON FUNCTIONS USED

nltk.download():
Downloads the required corpus.

brown.sents():
Loads Brown Corpus sentences.

reuters.sents():
Loads Reuters Corpus sentences.

Counter():
Counts word and N-gram frequencies.

set():
Stores unique vocabulary words.

update():
Adds counts to a Counter.

isalpha():
Checks whether a word contains only letters.

tuple():
Creates an ordered sequence of words.

len():
Returns the number of elements.

round():
Rounds decimal values.

============================================================

17. APPLICATIONS

1. News recommendation systems.
2. Medical text processing.
3. Financial language prediction.
4. Speech recognition.
5. Machine translation.
6. Chatbots.
7. Domain-specific autocomplete.

============================================================

CONCLUSION:

We trained a general language model using
the Brown Corpus and a domain-specific model
using the Reuters Corpus.

We applied Add-one smoothing to calculate
probabilities and interpolation to combine
both models.

This helps adapt a general language model
to a specific domain.

============================================================
"""
