
# !pip install -q nltk spacy pandas
# !python -m spacy download en_core_web_sm -q

import nltk
import spacy
import pandas as pd
from nltk.corpus import propbank

nltk.download("propbank", quiet=True)

nlp = spacy.load("en_core_web_sm")


# 1. DISPLAY DEPENDENCY PARSE

sentences = [
    "Riya opened the door with a key.",
    "The door was opened by Riya."
]

for sentence in sentences:
    doc = nlp(sentence)

    print("\nSentence:", sentence)

    for token in doc:
        print(token.text, "|", token.dep_, "| Head:", token.head.text)


# 2. EXTRACT COMPLETE PHRASES

def get_phrase(token, skip_token=False):

    words = [
        t.text for t in token.subtree
        if not (skip_token and t == token)
    ]

    return " ".join(words)


# 3. EXTRACT PREDICATE-ARGUMENT STRUCTURE

def extract_frames(doc):

    frames = []

    for verb in doc:

        if verb.pos_ != "VERB":
            continue

        frame = {"Predicate": verb.lemma_}

        for child in verb.children:

            dep = child.dep_.lower()

            # Agent in active sentence
            if dep == "nsubj":
                frame["ARG0 (Agent)"] = get_phrase(child)

            # Patient in passive sentence
            elif dep in ["nsubjpass", "nsubj:pass"]:
                frame["ARG1 (Patient)"] = get_phrase(child)

            # Direct object
            elif dep in ["dobj", "obj"]:
                frame["ARG1 (Patient)"] = get_phrase(child)

            # Agent in passive sentence
            elif dep in ["agent", "obl:agent"]:
                frame["ARG0 (Agent)"] = get_phrase(
                    child, skip_token=(child.text.lower() == "by")
                )

            # Recipient
            elif dep == "dative":
                frame["ARG2 (Recipient)"] = get_phrase(child)

            # Prepositional phrases
            elif dep == "prep":

                prep = child.text.lower()

                if prep == "by":
                    frame["ARG0 (Agent)"] = get_phrase(
                        child, skip_token=True
                    )

                elif prep == "with":
                    frame["ARG2 (Instrument)"] = get_phrase(
                        child, skip_token=True
                    )

                elif prep == "to":
                    frame["ARG2 (Recipient)"] = get_phrase(
                        child, skip_token=True
                    )

                else:
                    frame["Modifier [" + prep + "]"] = get_phrase(
                        child, skip_token=True
                    )

        frames.append(frame)

    return frames


# 4. ACTIVE AND PASSIVE SENTENCE COMPARISON

for sentence in sentences:

    print("\nSentence:", sentence)

    doc = nlp(sentence)

    for frame in extract_frames(doc):
        print(frame)


# 5. TEST MORE SENTENCES

more_sentences = [
    "Riya bought a car in Pune.",
    "A car was bought by Riya.",
    "The teacher gave the students a test.",
    "Arjun sent a letter to Meera.",
    "The thief broke the window with a hammer."
]

rows = []

for sentence in more_sentences:

    doc = nlp(sentence)

    for frame in extract_frames(doc):
        rows.append({"Sentence": sentence, **frame})

print("\nMORE PREDICATE-ARGUMENT FRAMES")

df = pd.DataFrame(rows).fillna("")
print(df.to_string(index=False))


# 6. DISPLAY PROPBANK ROLESETS

def show_roleset(role_id):

    roleset = propbank.roleset(role_id)

    print("\n", role_id)

    if roleset is None:
        print("Roleset not found.")
        return

    print("Verb:", roleset.attrib.get("name"))

    for role in roleset.findall("roles/role"):

        number = role.attrib["n"]
        description = role.attrib["descr"]

        print("ARG" + number, ":", description)


for role_id in ["open.01", "give.01", "buy.01", "break.01"]:
    show_roleset(role_id)


# 7. TEST DIFFICULT SENTENCES

hard_sentences = [
    "Arjun was given a book by Riya.",
    "Riya bought a car from Arjun.",
    "Riya opened the door and closed the window.",
    "The door opened.",
    "Riya saw the man with the telescope."
]

print("\nLIMITATIONS OF RULE-BASED EXTRACTION")

for sentence in hard_sentences:

    print("\nSentence:", sentence)

    doc = nlp(sentence)

    for frame in extract_frames(doc):
        print(frame)


"""
============================================================
       PRACTICAL 10: PREDICATE-ARGUMENT STRUCTURE
============================================================

1. WHAT IS SEMANTIC PARSING?

Semantic parsing means understanding the meaning of a
sentence and identifying its important parts.

It helps us understand:
- Who performed the action?
- What action was performed?
- What was affected?
- Who received something?
- What instrument was used?

============================================================

2. WHAT IS A PREDICATE?

A predicate represents an action or relation.
It is usually a verb.

Example:
Riya opened the door.

Predicate = opened

============================================================

3. WHAT IS AN ARGUMENT?

Arguments are the participants involved in an action.

Example:
Riya opened the door.

Agent = Riya
Patient = the door

============================================================

4. SEMANTIC ROLES

ARG0 = Agent
The person who performs the action.

ARG1 = Patient or Theme
The thing affected by the action.

ARG2 = Additional role, such as instrument
or recipient. Its exact meaning depends on the verb.

Example:
Riya opened the door with a key.

ARG0 = Riya
ARG1 = the door
ARG2 = a key (instrument)

============================================================

5. ACTIVE AND PASSIVE SENTENCES

Active:
Riya opened the door.

Passive:
The door was opened by Riya.

In the active sentence:
Riya = Agent
Door = Patient

In the passive sentence:
Riya = Agent
Door = Patient

The sentence structure changes, but the semantic
roles remain the same.

============================================================

6. WHAT IS DEPENDENCY PARSING?

Dependency parsing identifies grammatical
relationships between words.

Examples:

nsubj = Nominal subject
obj = Direct object
nsubjpass = Passive subject (older label)
nsubj:pass = Passive subject (newer label)
agent = Passive agent
prep = Preposition

============================================================

7. WHAT IS SPACY?

spaCy is a Python NLP library used for:
- Tokenization
- POS tagging
- Dependency parsing
- Named Entity Recognition

============================================================

8. WHAT IS PROPBANK?

PropBank means Proposition Bank.

It provides semantic role labels for verbs.

Example:
open.01

ARG0 = Agent
ARG1 = Thing opened
ARG2 = Instrument

The meaning of ARG2 can change depending on
the verb.

============================================================

9. WHAT IS SEMANTIC ROLE LABELING (SRL)?

SRL identifies the semantic roles of words
in a sentence.

It answers:
Who did what to whom, when, where and how?

============================================================

10. LIMITATIONS OF RULE-BASED EXTRACTION

1. It may confuse agent and patient.
2. It may fail with passive sentences.
3. It may miss shared subjects.
4. It may not understand ambiguous phrases.
5. Different verbs have different semantic roles.

============================================================

11. APPLICATIONS

1. Question answering systems
2. Chatbots
3. Information extraction
4. Machine translation
5. Text understanding
6. Search engines

============================================================

CONCLUSION:

We used spaCy dependency parsing to identify
predicates and their arguments.

We extracted semantic roles such as agent,
patient, instrument and recipient.

We also compared active and passive sentences
and understood the limitations of rule-based
semantic role extraction.

============================================================
"""
