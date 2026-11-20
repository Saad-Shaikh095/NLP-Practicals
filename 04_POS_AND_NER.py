
# !pip install -q nltk spacy scikit-learn seqeval pandas tabulate
# !python -m spacy download en_core_web_sm -q

import nltk
import spacy
import pandas as pd

from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from spacy.tokens import Doc
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as ner_report, f1_score
from seqeval.metrics.sequence_labeling import get_entities

for pkg in ["punkt_tab", "averaged_perceptron_tagger_eng",
            "maxent_ne_chunker_tab", "words", "brown", "universal_tagset"]:
    nltk.download(pkg, quiet=True)

nlp = spacy.load("en_core_web_sm")

sentence = "Alice works at Google in London."
tokens = word_tokenize(sentence)

# 1. POS Tagging
print("\nPOS TAGGING")

print("NLTK:", nltk.pos_tag(tokens))

doc = nlp(sentence)
print("spaCy:", [(t.text, t.tag_, t.pos_) for t in doc])

# 2. Named Entity Recognition
print("\nNAMED ENTITY RECOGNITION")

tree = nltk.ne_chunk(nltk.pos_tag(tokens))

print("NLTK:", tree)

print("NLTK Entities:", [
    (" ".join(w for w, tag in entity), entity.label())
    for entity in tree if isinstance(entity, Tree)
])

print("spaCy Entities:", [
    (entity.text, entity.label_)
    for entity in doc.ents
])

# 3. POS Tagging Evaluation
print("\nPOS TAGGING EVALUATION")

gold_sents = brown.tagged_sents(
    categories="news",
    tagset="universal"
)[:300]

spacy_map = {
    "PROPN": "NOUN",
    "AUX": "VERB",
    "CCONJ": "CONJ",
    "SCONJ": "CONJ",
    "PART": "PRT",
    "PUNCT": ".",
    "INTJ": "X",
    "SYM": "X",
    "SPACE": "X"
}

y_true = []
y_nltk = []
y_spacy = []

for sent in gold_sents:
    words = [word for word, tag in sent]

    y_true.extend([tag for word, tag in sent])

    y_nltk.extend([
        tag for word, tag in nltk.pos_tag(words, tagset="universal")
    ])

    doc = nlp(Doc(nlp.vocab, words=words))

    y_spacy.extend([
        spacy_map.get(token.pos_, token.pos_)
        for token in doc
    ])

def accuracy(actual, predicted):
    return sum(a == b for a, b in zip(actual, predicted)) / len(actual)

print("Total Tokens:", len(y_true))
print("NLTK Accuracy:", round(accuracy(y_true, y_nltk), 3))
print("spaCy Accuracy:", round(accuracy(y_true, y_spacy), 3))

print("\nNLTK Classification Report:")
print(classification_report(y_true, y_nltk, zero_division=0))

print("\nConfusion Matrix:")
print(pd.crosstab(
    pd.Series(y_true, name="Actual"),
    pd.Series(y_nltk, name="Predicted")
))

# 4. NLTK POS Errors
errors = pd.Series([
    (a, p) for a, p in zip(y_true, y_nltk) if a != p
]).value_counts().head(6)

print("\nCommon NLTK Errors:")
print(errors)

# 5. NER Evaluation
print("\nNER EVALUATION")

def make(tokens, spans):
    tags = ["O"] * len(tokens)

    for start, end, typ in spans:
        tags[start] = "B-" + typ

        for i in range(start + 1, end):
            tags[i] = "I-" + typ

    return tokens, tags

gold = [
    make("Alice works at Google in London .".split(), [(0,1,"PERSON"), (3,4,"ORGANIZATION"), (5,6,"GPE")]),
    make("Steve Jobs founded Apple in California .".split(), [(0,2,"PERSON"), (3,4,"ORGANIZATION"), (5,6,"GPE")]),
    make("Barack Obama visited Mumbai last year .".split(), [(0,2,"PERSON"), (3,4,"GPE")]),
    make("Microsoft opened a new office in Pune .".split(), [(0,1,"ORGANIZATION"), (6,7,"GPE")]),
    make("Sundar Pichai is the CEO of Google .".split(), [(0,2,"PERSON"), (6,7,"ORGANIZATION")]),
    make("Riya bought a car in Delhi .".split(), [(0,1,"PERSON"), (5,6,"GPE")]),
    make("The United Nations met in New York .".split(), [(1,3,"ORGANIZATION"), (5,7,"GPE")]),
    make("Tata Motors is based in Mumbai .".split(), [(0,2,"ORGANIZATION"), (5,6,"GPE")]),
    make("Priya studies at Stanford University in California .".split(), [(0,1,"PERSON"), (3,5,"ORGANIZATION"), (6,7,"GPE")]),
    make("Amazon hired engineers in Bangalore .".split(), [(0,1,"ORGANIZATION"), (4,5,"GPE")]),
    make("Narendra Modi met Joe Biden in Washington .".split(), [(0,2,"PERSON"), (3,5,"PERSON"), (6,7,"GPE")]),
    make("Infosys has offices in Pune and Hyderabad .".split(), [(0,1,"ORGANIZATION"), (4,5,"GPE"), (6,7,"GPE")])
]

keep = {"PERSON", "ORGANIZATION", "GPE"}

def nltk_predict(tokens):
    tree = nltk.ne_chunk(nltk.pos_tag(tokens))
    tags = []

    for child in tree:
        if isinstance(child, Tree):
            label = child.label()

            for i in range(len(child)):
                if label in keep:
                    tags.append(("B-" if i == 0 else "I-") + label)
                else:
                    tags.append("O")
        else:
            tags.append("O")

    return tags

def spacy_predict(tokens):
    doc = nlp(Doc(nlp.vocab, words=tokens))
    tags = []

    for token in doc:
        label = token.ent_type_

        if label == "ORG":
            label = "ORGANIZATION"

        if token.ent_iob_ == "O" or label not in keep:
            tags.append("O")
        else:
            tags.append(("B-" if token.ent_iob_ == "B" else "I-") + label)

    return tags

true_tags = [tags for tokens, tags in gold]
nltk_tags = [nltk_predict(tokens) for tokens, tags in gold]
spacy_tags = [spacy_predict(tokens) for tokens, tags in gold]

print("Total Sentences:", len(gold))

# 6. Token-wise Comparison
print("\nTOKEN-WISE COMPARISON")

for i in [0, 3, 6]:
    tokens, tags = gold[i]

    df = pd.DataFrame({
        "Token": tokens,
        "Gold": tags,
        "NLTK": nltk_tags[i],
        "spaCy": spacy_tags[i]
    })

    print("\n", df)

# 7. Entity-level Evaluation
print("\nNLTK ENTITY REPORT")
print(ner_report(true_tags, nltk_tags, zero_division=0))

print("\nSPACY ENTITY REPORT")
print(ner_report(true_tags, spacy_tags, zero_division=0))

# 8. Token-level Accuracy and Entity-level F1
def flatten(data):
    return [tag for sentence in data for tag in sentence]

print("\nFINAL COMPARISON")

print("NLTK Token Accuracy:",
      round(accuracy(flatten(true_tags), flatten(nltk_tags)), 3))

print("NLTK Entity F1:",
      round(f1_score(true_tags, nltk_tags), 3))

print("spaCy Token Accuracy:",
      round(accuracy(flatten(true_tags), flatten(spacy_tags)), 3))

print("spaCy Entity F1:",
      round(f1_score(true_tags, spacy_tags), 3))

# 9. Error Analysis
def show_errors(name, predictions):
    print("\n", name)

    for (tokens, tags), actual, predicted in zip(gold, true_tags, predictions):

        actual_entities = set(get_entities(actual))
        predicted_entities = set(get_entities(predicted))

        for typ, start, end in sorted(actual_entities - predicted_entities):
            print("MISSED:", " ".join(tokens[start:end+1]), typ)

        for typ, start, end in sorted(predicted_entities - actual_entities):
            print("EXTRA:", " ".join(tokens[start:end+1]), typ)

show_errors("NLTK ERRORS", nltk_tags)
show_errors("SPACY ERRORS", spacy_tags)


"""
=========================================================
EXPLANATION OF THE COMPLETE CODE
=========================================================

1. POS TAGGING:

POS stands for Part of Speech.

It identifies the grammatical category of each word.

Examples:
Noun       = Person, place or thing
Verb       = Action word
Adjective  = Describes a noun
Adverb     = Describes a verb or adjective
Pronoun    = Replaces a noun

Example:
Alice works at Google.

Alice  = Proper Noun
works  = Verb
at     = Preposition
Google = Proper Noun

We use:
nltk.pos_tag() for NLTK POS tagging.
nlp() for spaCy POS tagging.

2. NLTK VS SPACY POS TAGGING:

NLTK provides POS tags such as:
NN   = Noun
NNS  = Plural Noun
NNP  = Proper Noun
VB   = Verb
VBD  = Past Tense Verb
JJ   = Adjective
RB   = Adverb

spaCy provides:
tag_ = Detailed POS tag.
pos_ = General POS category.

3. NAMED ENTITY RECOGNITION (NER):

NER stands for Named Entity Recognition.

It identifies important names in a sentence.

Examples:
PERSON       = Alice, Rahul, Steve Jobs
ORGANIZATION = Google, Microsoft, Infosys
GPE          = India, London, Pune

Other common entity types include:
DATE  = 20 November 2026
MONEY = $1000
TIME  = 10 AM
NORP  = Nationalities or religious/political groups

We use:
nltk.ne_chunk() for NLTK NER.
doc.ents for spaCy NER.

4. NLTK NE_CHUNK:

ne_chunk() identifies named entities using
POS tags and creates a tree structure.

The tree contains:
- Named entity chunks.
- Individual words.

entity.label() returns the entity category.

5. SPACY NER:

spaCy identifies entities using its trained
English language model.

doc.ents returns all detected entities.

entity.text gives the entity name.
entity.label_ gives its category.

6. GOLD STANDARD:

A gold standard is a collection of manually
labelled data used to evaluate a model.

In this practical, we use the Brown corpus
for POS tagging evaluation.

The Brown corpus contains human-labelled text.

We compare the predicted POS tags with
the original correct tags.

7. POS TAGGING EVALUATION:

We evaluate NLTK and spaCy using the same words.

Accuracy formula:

Accuracy = Correct Predictions / Total Predictions

Example:

Correct predictions = 90
Total predictions = 100

Accuracy = 90/100 = 90%

8. CONFUSION MATRIX:

A confusion matrix compares actual labels
with predicted labels.

Rows represent actual tags.
Columns represent predicted tags.

It helps identify which POS categories
are commonly confused.

9. CLASSIFICATION REPORT:

classification_report() provides:

Precision = Correct positive predictions /
            Total positive predictions.

Recall = Correct positive predictions /
         Total actual positive values.

F1 Score = Harmonic mean of Precision and Recall.

Support = Number of actual examples in each class.

10. IOB2 FORMAT:

IOB2 is used to identify the boundaries
of named entities.

B = Beginning of an entity.
I = Inside an entity.
O = Outside an entity.

Example:

Steve Jobs works at Google.

Steve = B-PERSON
Jobs  = I-PERSON
works = O
at    = O
Google = B-ORGANIZATION

11. GOLD NER DATA:

The make() function creates manually labelled
NER data.

It takes:
- Tokens
- Starting position
- Ending position
- Entity type

It creates B-, I- and O- tags.

Entity types used:
PERSON
ORGANIZATION
GPE

12. NLTK NER PREDICTION:

nltk_predict() performs NER using NLTK.

It first performs POS tagging.

Then ne_chunk() identifies named entities.

Finally, it converts entities into IOB2 format.

13. SPACY NER PREDICTION:

spacy_predict() performs NER using spaCy.

It uses:
ent_type_ = Entity category.
ent_iob_ = Entity position.

It converts spaCy labels into the same
IOB2 format used by the gold data.

ORG is converted into ORGANIZATION
so both tools use the same entity labels.

14. TOKEN-WISE COMPARISON:

It compares each token's:
- Actual Gold tag.
- NLTK predicted tag.
- spaCy predicted tag.

This helps identify incorrect predictions.

15. ENTITY-LEVEL EVALUATION:

NER should be evaluated at the entity level.

An entity is correct only when:
- Its complete boundary is correct.
- Its entity type is correct.

seqeval is used for entity-level evaluation.

It provides precision, recall and F1 score.

16. TOKEN-LEVEL ACCURACY VS ENTITY F1:

Token accuracy measures individual token
predictions.

Entity F1 measures complete entity predictions.

Token accuracy can be misleading because
most words are usually labelled O.

Example:

Sentence:
Alice works at Google.

If most tokens are correctly labelled O,
token accuracy can be high even when
Alice or Google is identified incorrectly.

Therefore, entity-level F1 is important
for proper NER evaluation.

17. ERROR ANALYSIS:

Error analysis identifies the entities
that the model missed or predicted incorrectly.

MISSED = Gold entity was not predicted correctly.

EXTRA = Model predicted an entity that
         does not match the gold data.

This helps us understand model weaknesses.

18. APPLICATIONS OF POS TAGGING:

- Grammar checking
- Sentence analysis
- Language translation
- Text processing
- Question answering

19. APPLICATIONS OF NER:

- Information extraction
- Search engines
- Chatbots
- Resume analysis
- News analysis
- Medical information extraction

=========================================================
IMPORTANT PYTHON FUNCTIONS
=========================================================

word_tokenize()       - Splits sentences into words.
pos_tag()             - Performs POS tagging.
ne_chunk()            - Performs NLTK NER.
spacy.load()          - Loads spaCy language model.
nlp()                 - Processes text using spaCy.
doc.ents              - Returns named entities.
entity.label()        - Returns NLTK entity label.
entity.label_         - Returns spaCy entity label.
classification_report() - Evaluates POS classification.
crosstab()            - Creates a confusion table.
accuracy()            - Calculates accuracy.
f1_score()            - Calculates entity F1 score.
get_entities()        - Extracts entity spans.
extend()              - Adds multiple elements to a list.
flatten()             - Converts nested lists into one list.

=========================================================
END OF EXPERIMENT
=========================================================
"""
