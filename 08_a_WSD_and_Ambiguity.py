
# !pip install -q nltk pandas

import nltk
import pandas as pd
from nltk import word_tokenize, CFG
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
from nltk.parse import ChartParser

for pkg in ["wordnet", "omw-1.4", "punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)


# 1. EXPLORE WORDNET SENSES

print("MEANINGS OF BANK")

for i, syn in enumerate(wn.synsets("bank", pos=wn.NOUN), 1):
    print(i, syn.name(), "-", syn.definition())


# 2. TEST DATA

test = [
    ("I deposited money in the bank.", "FIN"),
    ("She opened a bank account.", "FIN"),
    ("The bank approved my loan application.", "FIN"),
    ("The bank charges a fee for every transaction.", "FIN"),
    ("He sat on the bank of the river.", "RIV"),
    ("The children played near the river bank.", "RIV"),
    ("We had a picnic on the grassy bank.", "RIV"),
    ("Fishing from the bank of the lake is relaxing.", "RIV"),
    ("The river overflowed its bank after heavy rain.", "RIV"),
    ("I went to the bank.", "?")
]

FIN_SYNSETS = {
    "depository_financial_institution.n.01",
    "bank.n.09"
}

RIV_SYNSETS = {"bank.n.01"}

def coarse(syn):
    if syn is None:
        return "?"
    if syn.name() in FIN_SYNSETS:
        return "FIN"
    if syn.name() in RIV_SYNSETS:
        return "RIV"
    return "OTHER"


# 3. METHOD A: KEYWORD RULES

def rule_v1(sentence):
    s = sentence.lower()

    if any(w in s for w in ["money", "deposit", "account", "loan"]):
        return "FIN"

    if any(w in s for w in ["river", "water", "shore", "fishing"]):
        return "RIV"

    return "?"


print("\nKEYWORD RULE EXAMPLES")
print(rule_v1("I deposited money in the bank."))
print(rule_v1("He sat near the river bank."))


# 4. METHOD B: FIRST-SENSE BASELINE

first = wn.synsets("bank", pos=wn.NOUN)[0]

print("\nFIRST SENSE")
print(first.name(), "-", first.definition())

def first_sense(sentence):
    return coarse(first)


# 5. METHOD C: NLTK LESK ALGORITHM

def nltk_lesk(sentence):
    syn = lesk(word_tokenize(sentence), "bank", pos="n")
    return coarse(syn)


print("\nLESK EXAMPLE")

for s, label in test:
    syn = lesk(word_tokenize(s), "bank", pos="n")
    print(s, "->", syn.name(), "-", syn.definition())


# 6. METHOD D: IMPROVED GLOSS OVERLAP

STOP = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def bag(text):
    return {
        lemm.lemmatize(w.lower())
        for w in word_tokenize(text)
        if w.isalpha() and w.lower() not in STOP
    }


def signature(syn, expand=True):
    text = syn.definition() + " " + " ".join(syn.examples())

    if expand:
        related = (
            syn.hypernyms() +
            syn.hyponyms() +
            syn.part_meronyms() +
            syn.member_holonyms()
        )

        for r in related:
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

    return bag(text)


CANDIDATES = {
    "RIV": wn.synset("bank.n.01"),
    "FIN": wn.synset("depository_financial_institution.n.01")
}


def overlap_wsd(sentence, expand=True, word="bank"):
    context = bag(sentence) - {word}

    scores = {
        label: len(context & signature(syn, expand))
        for label, syn in CANDIDATES.items()
    }

    best = sorted(scores.items(), key=lambda x: -x[1])

    if best[0][1] == 0 or best[0][1] == best[1][1]:
        return "?"

    return best[0][0]


print("\nIMPROVED WSD EXAMPLE")
print(overlap_wsd("I deposited money in the bank."))


# 7. COMPARE ALL METHODS

methods = {
    "Keyword Rules": rule_v1,
    "First Sense": first_sense,
    "NLTK Lesk": nltk_lesk,
    "Overlap Without Expansion": lambda s: overlap_wsd(s, False),
    "Overlap With Expansion": overlap_wsd
}

labelled = [(s, g) for s, g in test if g != "?"]

rows = []

for name, method in methods.items():
    predictions = [method(s) for s, g in labelled]

    correct = sum(
        p == g for p, (_, g) in zip(predictions, labelled)
    )

    abstain = predictions.count("?")
    wrong = len(labelled) - correct - abstain

    rows.append({
        "Method": name,
        "Correct": f"{correct}/{len(labelled)}",
        "Abstained": abstain,
        "Wrong": wrong
    })

print("\nMETHOD COMPARISON")
print(pd.DataFrame(rows).to_string(index=False))


# 8. TEST IMPROVED WSD ON ALL SENTENCES

print("\nIMPROVED WSD PREDICTIONS")

for s, g in test:
    print("Gold:", g, "| Predicted:", overlap_wsd(s), "|", s)


# 9. LESK ON OTHER AMBIGUOUS WORDS

examples = [
    ("bat", "The bat flew out of the cave at night."),
    ("bat", "He hit the ball with a bat."),
    ("plant", "The plant manufactures car parts."),
    ("plant", "Water the plant every day.")
]

print("\nOTHER WORD SENSE EXAMPLES")

for word, sentence in examples:
    syn = lesk(word_tokenize(sentence), word, pos="n")
    print(word, "->", syn.name(), "-", syn.definition())


# 10. STRUCTURAL AMBIGUITY

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N | NP PP | 'I'
VP -> V NP | VP PP
PP -> P NP
Det -> 'the'
N -> 'man' | 'telescope'
V -> 'saw'
P -> 'with'
""")

parser = ChartParser(grammar)

sentence = "I saw the man with the telescope"

trees = list(parser.parse(sentence.split()))

print("\nSTRUCTURAL AMBIGUITY")
print("Number of parse trees:", len(trees))

for tree in trees:
    tree.pretty_print()


"""
============================================================
                 PRACTICAL 9A EXPLANATION
============================================================

TOPIC:
Word Sense Disambiguation and Ambiguity Resolution.

1. WHAT IS AMBIGUITY?

Ambiguity means a word or sentence has more than one meaning.

Example:
Bank can mean:
- Financial institution.
- Land beside a river.

2. WHAT IS WORD SENSE DISAMBIGUATION (WSD)?

WSD is the process of identifying the correct meaning
of a word based on the surrounding words or context.

Example:
"I deposited money in the bank."

Here, bank means a financial institution.

"He sat near the river bank."

Here, bank means land beside a river.

3. WHAT IS WORDNET?

WordNet is a lexical database of English words.

It groups words with similar meanings into synsets.

Synset = Set of synonyms representing one word sense.

Important functions:

wn.synsets():
Returns possible meanings of a word.

wn.synset():
Accesses a particular synset.

definition():
Returns the meaning of a synset.

lemma_names():
Returns synonyms or word forms in a synset.

examples():
Returns example sentences.

============================================================

4. METHOD A: KEYWORD RULES

Definition:
This method identifies word meaning using predefined
keywords.

Example:
money, deposit, account, loan -> FIN
river, water, shore, fishing -> RIV

Advantages:
- Simple.
- Easy to implement.
- Fast for known keywords.

Disadvantages:
- Needs manually written rules.
- May fail when keywords are missing.
- Can give wrong results when both types of words occur.

============================================================

5. METHOD B: FIRST-SENSE BASELINE

Definition:
This method always selects the first meaning of a word
listed in WordNet.

It is called a baseline because other WSD methods
are compared against it.

Limitation:
It does not understand the sentence context.

============================================================

6. METHOD C: LESK ALGORITHM

Definition:
Lesk selects the word meaning whose dictionary definition
has the greatest word overlap with the sentence.

Example:
"I deposited money in the bank."

It compares the sentence words with the definitions
of different bank meanings.

More matching words indicate a better sense.

Limitation:
- Definitions may be short.
- It may select an incorrect meaning.
- Common words may create accidental matches.

============================================================

7. METHOD D: IMPROVED GLOSS OVERLAP

This method improves Lesk using three techniques:

1. Stop-word removal:
Removes common words like the, is, and of.

2. Lemmatization:
Converts words into their base form.

Example:
running -> running (or run when POS-aware)
deposited -> deposited (or deposit when POS-aware)

3. Gloss expansion:
Uses definitions and examples of related WordNet senses.

4. Abstaining:
Returns ? when there is no clear match or both senses
receive the same score.

Important functions:

word_tokenize():
Splits text into tokens.

stopwords.words():
Returns common stop words.

WordNetLemmatizer():
Performs lemmatization.

hypernyms():
Returns more general concepts.

hyponyms():
Returns more specific concepts.

part_meronyms():
Returns parts of something.

member_holonyms():
Returns a whole whose members include the given concept.

============================================================

8. METHOD COMPARISON

Keyword Rules:
Uses predefined keywords.

First Sense:
Always selects the first WordNet meaning.

Lesk:
Compares sentence words with dictionary definitions.

Improved Gloss Overlap:
Removes stop words, lemmatizes and expands definitions.

Correct:
Number of correct predictions.

Abstained:
Number of times the method returns ?.

Wrong:
Number of incorrect predictions.

============================================================

9. LEXICAL VS STRUCTURAL AMBIGUITY

Lexical Ambiguity:
One word has multiple meanings.

Example:
Bank = financial institution or river side.

Structural Ambiguity:
One sentence has multiple grammatical structures.

Example:
"I saw the man with the telescope."

Meaning 1:
I used a telescope to see the man.

Meaning 2:
The man had a telescope.

10. CONTEXT-FREE GRAMMAR

S -> NP VP
NP -> Det N | NP PP | I
VP -> V NP | VP PP
PP -> P NP

ChartParser generates possible parse trees.

If there are two parse trees, the sentence has
two possible grammatical structures under this grammar.

============================================================

11. IMPORTANT FUNCTIONS

enumerate():
Provides index and value.

set():
Stores unique elements.

len():
Returns the number of elements.

any():
Checks whether at least one condition is True.

sorted():
Sorts elements.

Counter:
Not used in this practical.

pd.DataFrame():
Creates a table of results.

============================================================

12. APPLICATIONS

1. Search engines.
2. Machine translation.
3. Chatbots.
4. Question answering.
5. Text understanding.
6. Information extraction.
7. Sentiment analysis.

Modern WSD systems often use contextual language models
such as BERT to understand word meanings.

============================================================

CONCLUSION:

WSD identifies the correct meaning of ambiguous words
using sentence context. WordNet provides possible meanings,
while keyword rules, first-sense baseline, Lesk and improved
gloss overlap are different methods of selecting a sense.

Structural ambiguity occurs when a sentence has multiple
possible grammatical structures.
============================================================
"""
