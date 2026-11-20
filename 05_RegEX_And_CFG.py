
# !pip install -q nltk

import re
import nltk
from nltk import CFG
from nltk.parse import ChartParser

DET = ["the", "a", "an"]
NOUN = ["cat", "dog", "student", "teacher", "boy", "girl", "book", "food", "cricket"]
VERB = ["eats", "likes", "reads", "writes", "plays", "teaches", "buys"]

def alt(words):
    return "(?:" + "|".join(words) + ")"

regex_pattern = rf"{alt(DET)}\s+{alt(NOUN)}\s+{alt(VERB)}\s+{alt(NOUN)}"

def clean(sentence):
    return re.sub(r"[.!?]$", "", sentence.strip().lower())

def regex_validate(sentence):
    return bool(re.fullmatch(regex_pattern, clean(sentence)))

def quote(words):
    return " | ".join(repr(w) for w in words)

grammar = CFG.fromstring(f"""
S -> NP VP
NP -> Det N | N
VP -> V NP
Det -> {quote(DET)}
N -> {quote(NOUN)}
V -> {quote(VERB)}
""")

parser = ChartParser(grammar)
KNOWN = set(DET + NOUN + VERB)

def cfg_validate(sentence):
    words = clean(sentence).split()
    unknown = [w for w in words if w not in KNOWN]

    if unknown:
        return False, "Unknown words: " + str(unknown), []

    trees = list(parser.parse(words))

    if not trees:
        return False, "Incorrect sentence structure", []

    return True, "OK", trees

sentence = "The cat eats food"

ok_regex = regex_validate(sentence)
ok_cfg, reason, trees = cfg_validate(sentence)

print("Sentence:", sentence)
print("Regex:", "VALID" if ok_regex else "INVALID")
print("CFG:", "VALID" if ok_cfg else "INVALID", reason)

if trees:
    print(trees[0])
    trees[0].pretty_print()

tests = [
    "The cat eats food",
    "The cat eats the food",
    "Cat eats food",
    "The cat eats",
    "Eats the cat food",
    "The cat flies food",
    "The food eats cat"
]

print("\nSentence                       Regex       CFG")

for s in tests:
    r = regex_validate(s)
    c, _, _ = cfg_validate(s)

    print(f"{s:30} {'VALID' if r else 'INVALID':10} {'VALID' if c else 'INVALID'}")

recursive = CFG.fromstring(f"""
S -> NP VP
NP -> Det N | N | NP PP
PP -> P NP
VP -> V NP
Det -> {quote(DET)}
N -> {quote(NOUN + ['mat', 'door'])}
V -> {quote(VERB)}
P -> 'on' | 'near'
""")

rparser = ChartParser(recursive)

sentences = [
    "the cat eats food",
    "the cat on the mat eats food",
    "the cat on the mat near the door eats food"
]

for s in sentences:
    trees = list(rparser.parse(s.split()))
    print("\nSentence:", s)
    print("Number of parse trees:", len(trees))

trees = list(rparser.parse("the cat on the mat near the door eats food".split()))

if trees:
    trees[0].pretty_print()

plural = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the'
N -> 'cat' | 'cats' | 'food'
V -> 'eats' | 'eat'
""")

plural_parser = ChartParser(plural)

for s in ["the cat eats the food", "the cats eats the food"]:
    result = list(plural_parser.parse(s.split()))
    print(s, "->", "VALID" if result else "INVALID")


"""
============================================================
                 PRACTICAL 5 EXPLANATION
============================================================

TOPIC:
Regular Expressions, Context-Free Grammar and Constituency Parsing.

1. REGULAR EXPRESSION (REGEX)
Regex is a pattern used to search, match or validate text.
Here, regex checks this sentence structure:

Determiner + Noun + Verb + Noun

Example:
The cat eats food.

2. IMPORTANT REGEX FUNCTIONS

re.fullmatch():
Checks whether the complete sentence matches the pattern.

re.sub():
Replaces or removes matching text.
Here, it removes punctuation from the end.

strip():
Removes extra spaces from the beginning and end.

lower():
Converts all letters into lowercase.

bool():
Returns True or False.

3. CONTEXT-FREE GRAMMAR (CFG)
CFG is a set of grammar rules used to describe sentence structure.

Our grammar:

S -> NP VP
NP -> Det N | N
VP -> V NP

Meaning:

S  = Sentence
NP = Noun Phrase
VP = Verb Phrase
Det = Determiner
N = Noun
V = Verb

Example:
The cat eats food.

S
├── NP
│   ├── Det -> The
│   └── N -> cat
└── VP
    ├── V -> eats
    └── NP
        └── N -> food

4. CHART PARSER
ChartParser is an NLTK parser that checks whether a sentence
follows the given CFG rules.

parser.parse():
Generates possible parse trees.

pretty_print():
Displays the parse tree in a readable format.

5. REGEX VS CFG

Regex:
- Checks a flat sequence of words.
- Works well for fixed patterns.
- Does not naturally represent hierarchical sentence structure.

CFG:
- Uses grammar production rules.
- Represents phrases inside sentences.
- Can generate hierarchical parse trees.
- Supports recursive grammar rules.

6. RECURSION
Recursion means a grammar rule can refer to itself.

Example:
NP -> NP PP

PP means Prepositional Phrase.

This allows sentences like:
The cat on the mat eats food.
The cat on the mat near the door eats food.

7. STRUCTURAL AMBIGUITY
One sentence can have multiple possible parse trees.

For example:
The cat on the mat near the door eats food.

The phrase "near the door" may describe the mat
or the cat.

This is called structural ambiguity.

8. GRAMMAR AGREEMENT
A basic CFG does not automatically understand singular
and plural agreement.

For example:
The cats eats food.

Our grammar may accept this because it does not
check whether the subject and verb agree.

============================================================
IMPORTANT FORMULAS / RULES
============================================================

Regex pattern:
Determiner + Noun + Verb + Noun

CFG:
S -> NP VP
NP -> Det N | N
VP -> V NP

Recursive rule:
NP -> NP PP

============================================================
APPLICATIONS
============================================================

1. Sentence validation.
2. Text processing.
3. Natural Language Understanding.
4. Grammar checking.
5. Machine translation.
6. Syntax analysis.
7. Constituency parsing.

============================================================
CONCLUSION
============================================================

Regex validates fixed word patterns, while CFG describes
hierarchical sentence structure using grammar rules.
NLTK ChartParser generates parse trees to understand
how words are grouped into phrases.
============================================================
"""
