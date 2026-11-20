
# !pip install -q nltk spacy pandas
# !python -m spacy download en_core_web_sm -q

import nltk
import spacy
import pandas as pd
from nltk import CFG
from nltk.parse import ChartParser
from nltk.corpus import treebank
from collections import Counter
from spacy import displacy

nltk.download("treebank", quiet=True)

nlp = spacy.load("en_core_web_sm")

# PROGRAM 1: DEPENDENCY PARSING

text = "The cat sat on the mat."
doc = nlp(text)

data = []

for t in doc:
    data.append({
        "Word": t.text,
        "POS": t.pos_,
        "Dependency": t.dep_,
        "Head": t.head.text,
        "Meaning": spacy.explain(t.dep_)
    })

df = pd.DataFrame(data)
print("DEPENDENCY PARSING")
print(df)

displacy.render(doc, style="dep", jupyter=True)

print("\nDEPENDENCY TREE")

for t in doc:
    print(t.text, "<- Head:", t.head.text,
          "| Children:", [c.text for c in t.children])


# PROGRAM 2: CONSTITUENCY PARSING

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V PP
PP -> P NP
Det -> 'The' | 'the'
N -> 'cat' | 'mat'
V -> 'sat'
P -> 'on'
""")

parser = ChartParser(grammar)

def parse(sentence):
    words = sentence.rstrip(".!?").split()

    try:
        trees = list(parser.parse(words))
    except ValueError as err:
        print("Cannot parse:", err)
        return []

    if not trees:
        print("No parse found")

    for tree in trees:
        print(tree)
        tree.pretty_print()

    return trees

print("\nCONSTITUENCY PARSING")

parse("The cat sat on the mat.")

print("\nUNKNOWN WORD EXAMPLE")

parse("The dog sat on the mat.")


# PROGRAM 3: TREEBANK

print("\nTREEBANK")

print("Number of sentences:", len(treebank.sents()))

print("\nSample Sentence:")
print(" ".join(treebank.sents()[0]))

print("\nPOS Tagged Sentence:")
print(treebank.tagged_sents()[0])

tree = treebank.parsed_sents()[0]

print("\nTREEBANK PARSE TREE")
print(tree)
tree.pretty_print()


# EXTRACT COMMON GRAMMAR RULES

rules = Counter()

for t in treebank.parsed_sents()[:200]:
    for production in t.productions():
        if production.is_nonlexical():
            rules[str(production)] += 1

print("\nMOST COMMON GRAMMAR RULES")

for rule, count in rules.most_common(8):
    print(count, rule)


# DEPENDENCY PARSING OF TREEBANK SENTENCE

sent = " ".join(treebank.sents()[0])
d = nlp(sent)

print("\nSPACY DEPENDENCY PARSING")

data = []

for t in d:
    data.append({
        "Word": t.text,
        "Dependency": t.dep_,
        "Head": t.head.text
    })

print(pd.DataFrame(data))


"""
============================================================
                 PRACTICAL 6 EXPLANATION
============================================================

TOPIC:
Dependency Parsing and Syntactic Treebanks.

PROGRAM 1: DEPENDENCY PARSING

1. What is Dependency Parsing?

Dependency parsing identifies the grammatical relationship
between words in a sentence.

Example:
The cat sat on the mat.

It identifies:
- Main action = sat
- Who performed the action = cat
- Where = on the mat

2. IMPORTANT TERMS

Head:
The main word on which another word depends.

Dependent:
A word that is connected to another word.

ROOT:
The main word of the sentence.
Usually, it is the main verb.

Children:
Words that directly depend on a particular word.

3. IMPORTANT FUNCTIONS

spacy.load():
Loads the trained English language model.

nlp(text):
Processes the sentence using spaCy.

t.text:
Returns the actual word.

t.pos_:
Returns the part of speech.

t.dep_:
Returns the dependency relation.

t.head:
Returns the head word.

t.children:
Returns the dependent words.

spacy.explain():
Explains a dependency label.

displacy.render():
Displays the dependency tree visually.

4. COMMON DEPENDENCY LABELS

ROOT = Main word of the sentence.
nsubj = Nominal subject.
dobj = Direct object (legacy label).
obj = Object in modern spaCy models.
det = Determiner.
prep = Preposition (legacy label).
pobj = Object of preposition (legacy label).
ADP = Adposition POS category.

Note:
The exact dependency labels depend on the spaCy model.
Modern English spaCy models generally use labels such as
obj and pobj-related structures may differ.

============================================================

PROGRAM 2: CONSTITUENCY PARSING

1. What is Constituency Parsing?

It divides a sentence into grammatical groups called phrases.

Example:
The cat sat on the mat.

It divides the sentence into:
- NP = Noun Phrase
- VP = Verb Phrase
- PP = Prepositional Phrase

2. CFG RULES

S -> NP VP
NP -> Det N
VP -> V PP
PP -> P NP

Meaning:

S = Sentence
NP = Noun Phrase
VP = Verb Phrase
PP = Prepositional Phrase
Det = Determiner
N = Noun
V = Verb
P = Preposition

3. IMPORTANT FUNCTIONS

CFG.fromstring():
Creates a grammar from production rules.

ChartParser():
Creates a parser using the grammar.

parser.parse():
Checks the sentence and generates parse trees.

pretty_print():
Displays the parse tree in a readable format.

rstrip():
Removes specified characters from the right side
of a string.

try-except:
Handles errors when a word is not present in grammar.

============================================================

PROGRAM 3: TREEBANK

1. What is a Treebank?

A treebank is a collection of sentences annotated
with grammatical structures or parse trees.

It is used to train and evaluate NLP parsers.

2. PENN TREEBANK

Penn Treebank is a famous collection of linguistically
annotated English sentences.

NLTK provides a sample of 3,914 sentences from it.

3. IMPORTANT FUNCTIONS

treebank.sents():
Returns sentences.

treebank.tagged_sents():
Returns sentences with POS tags.

treebank.parsed_sents():
Returns sentences with parse trees.

productions():
Extracts grammar production rules from a tree.

Counter():
Counts how frequently each grammar rule occurs.

most_common():
Returns the most frequently occurring rules.

4. IMPORTANT TREEBANK LABELS

S = Sentence
NP = Noun Phrase
VP = Verb Phrase
PP = Prepositional Phrase
ADJP = Adjective Phrase

Function tags:
-SBJ = Subject
-TMP = Time
-CLR = Closely related to the verb
-PRD = Predicate

POS tags:
NNP = Proper Noun
CD = Cardinal Number
MD = Modal
VB = Base Verb
DT = Determiner
IN = Preposition or subordinating conjunction

-NONE-:
Represents empty elements such as traces or omitted
syntactic elements in the Penn Treebank.

============================================================

DEPENDENCY VS CONSTITUENCY PARSING

Dependency Parsing:
- Connects words directly.
- Focuses on head and dependent relationships.
- Useful for extracting subjects and objects.

Constituency Parsing:
- Groups words into phrases.
- Uses labels such as NP, VP and PP.
- Useful for understanding sentence structure.

============================================================

APPLICATIONS

1. Machine Translation.
2. Grammar checking.
3. Question Answering.
4. Information Extraction.
5. Sentiment Analysis.
6. Natural Language Understanding.
7. Text summarization.

============================================================

CONCLUSION

Dependency parsing identifies relationships between words.
Constituency parsing divides sentences into grammatical
phrases. Treebanks provide real annotated sentences that
help train and evaluate NLP parsing systems.
============================================================
"""
