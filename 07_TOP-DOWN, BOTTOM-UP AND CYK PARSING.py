
# PRACTICAL 7: TOP-DOWN, BOTTOM-UP AND CYK PARSING

sentence = input("Enter a sentence: ").lower().split()


# 1. TOP-DOWN PARSING

class TopDownParser:
    def __init__(self, grammar):
        self.grammar = grammar

    def parse(self, tokens):
        self.tokens = tokens
        return self.match("S", 0) == len(tokens)

    def match(self, symbol, index):
        if symbol not in self.grammar:
            if index < len(self.tokens) and self.tokens[index] == symbol:
                return index + 1
            return -1

        for expansion in self.grammar[symbol]:
            current = index
            for part in expansion:
                current = self.match(part, current)
                if current == -1:
                    break
            else:
                return current

        return -1


grammar = {
    "S": [["NP", "VP"]],
    "NP": [["Det", "N"]],
    "VP": [["V", "NP"], ["V"]],
    "Det": [["the"]],
    "N": [["cat"], ["mouse"]],
    "V": [["chased"]]
}

parser = TopDownParser(grammar)

print("\nTop-Down Parsing:",
      parser.parse(sentence))


# 2. BOTTOM-UP SHIFT-REDUCE PARSING

class ShiftReduceParser:
    def __init__(self, rules):
        self.rules = rules

    def parse(self, tokens):
        stack = []
        remaining = list(tokens)

        while remaining or stack:
            reduced = False

            for length in range(len(stack), 0, -1):
                suffix = tuple(stack[-length:])

                if suffix in self.rules:
                    stack = stack[:-length] + [self.rules[suffix]]
                    reduced = True
                    break

            if not reduced and remaining:
                stack.append(remaining.pop(0))

            elif not reduced:
                break

        return stack == ["S"]


rules = {
    ("the",): "Det",
    ("cat",): "N",
    ("mouse",): "N",
    ("chased",): "V",
    ("Det", "N"): "NP",
    ("V", "NP"): "VP",
    ("NP", "VP"): "S"
}

parser = ShiftReduceParser(rules)

print("Bottom-Up Parsing:",
      parser.parse(sentence))


# 3. CHART PARSING - CYK ALGORITHM

class CYKParser:
    def __init__(self, grammar):
        self.grammar = grammar

    def parse(self, tokens):
        n = len(tokens)

        if n == 0:
            return False

        table = [[set() for _ in range(n + 1)]
                 for _ in range(n + 1)]

        # Step 1: Find grammar rules for individual words
        for i, word in enumerate(tokens):
            for left, rights in self.grammar.items():
                for right in rights:
                    if len(right) == 1 and right[0] == word:
                        table[1][i + 1].add(left)

        # Step 2: Combine smaller phrases into bigger phrases
        for length in range(2, n + 1):
            for start in range(1, n - length + 2):
                for split in range(1, length):

                    for left, rights in self.grammar.items():
                        for right in rights:
                            if len(right) == 2:
                                B, C = right

                                if (B in table[split][start] and
                                    C in table[length - split][start + split]):
                                    table[length][start].add(left)

        print("\nCYK Parsing Table:")

        for length in range(1, n + 1):
            print("Length", length, ":", table[length][1:n-length+2])

        return "S" in table[n][1]


cnf_grammar = {
    "S": [["NP", "VP"]],
    "NP": [["Det", "N"]],
    "VP": [["V", "NP"]],
    "Det": [["the"]],
    "N": [["cat"], ["mouse"]],
    "V": [["chased"]]
}

parser = CYKParser(cnf_grammar)

print("CYK Chart Parsing:",
      parser.parse(sentence))


"""
============================================================
                 PRACTICAL 7 EXPLANATION
============================================================

TOPIC:
Top-Down, Bottom-Up and Chart/CYK Parsing Algorithms.

COMMON GRAMMAR USED:

S -> NP VP
NP -> Det N
VP -> V NP
VP -> V
Det -> the
N -> cat | mouse
V -> chased

Example sentence:
The cat chased the mouse.

S = Sentence
NP = Noun Phrase
VP = Verb Phrase
Det = Determiner
N = Noun
V = Verb

============================================================

1. TOP-DOWN PARSING

Definition:
Top-Down Parsing starts from the starting symbol S
and tries to generate the input sentence.

It moves from the general grammar structure
towards individual words.

Steps:
1. Start with S.
2. Expand S into NP and VP.
3. Expand NP into Det and N.
4. Expand VP into V and NP.
5. Match the words with the input sentence.
6. If all words match, sentence is VALID.

Important functions:

__init__():
Initializes the grammar.

parse():
Starts parsing from S.

match():
Matches grammar symbols with input words.

Base case:
If the symbol is a terminal word, compare it
with the current input word.

Recursive case:
If the symbol is a non-terminal, try its
production rules.

============================================================

2. BOTTOM-UP PARSING (SHIFT-REDUCE)

Definition:
Bottom-Up Parsing starts from individual input words
and combines them into larger grammatical structures.

It moves from words towards the starting symbol S.

Two main operations:

SHIFT:
Moves the next input word onto the stack.

REDUCE:
Combines words or grammar symbols into a
larger grammar symbol.

Example:
The cat chased the mouse.

SHIFT:
the -> cat -> chased -> the -> mouse

REDUCE:
the -> Det
cat -> N
Det + N -> NP
chased -> V
the -> Det
mouse -> N
Det + N -> NP
V + NP -> VP
NP + VP -> S

If the final stack contains S, the sentence is VALID.

Important concepts:

Stack:
Stores the words and grammar symbols.

Remaining:
Stores the words that have not been shifted.

Rules:
Contains the combinations that can be reduced.

============================================================

3. CHART PARSING - CYK ALGORITHM

Definition:
CYK stands for Cocke-Younger-Kasami.

It is a dynamic programming parsing algorithm.
It uses a table to find grammatical structures.

Important condition:
The grammar must be in Chomsky Normal Form (CNF).

CNF RULES:

A -> B C
A -> word

Where:
A, B, C are non-terminal symbols.
word is a terminal symbol.

Steps:

1. Create an empty parsing table.
2. Match individual words with terminal rules.
3. Store the matching grammar symbols.
4. Combine smaller phrases using grammar rules.
5. Continue until the complete sentence is checked.
6. If S appears in the top cell, the sentence is VALID.

Important variables:

n = Number of input words.
table = Stores possible grammar symbols.
length = Length of the current word span.
start = Starting position of the span.
split = Position where the span is divided.

Important functions:

enumerate():
Returns index and value together.

set():
Stores unique grammar symbols.

len():
Returns the number of elements.

============================================================

4. DIFFERENCE BETWEEN THREE ALGORITHMS

TOP-DOWN:
- Starts from S.
- Breaks grammar into smaller parts.
- Works from grammar to words.
- Uses recursive matching.

BOTTOM-UP:
- Starts from input words.
- Combines words into phrases.
- Works from words to S.
- Uses stack, SHIFT and REDUCE.

CYK:
- Uses dynamic programming.
- Uses a two-dimensional table.
- Combines smaller phrases into larger phrases.
- Requires CNF grammar.

============================================================

5. APPLICATIONS

1. Natural Language Processing.
2. Syntax analysis.
3. Compiler design.
4. Grammar checking.
5. Machine translation.
6. Sentence validation.
7. Natural Language Understanding.

============================================================

CONCLUSION:

Top-Down parsing starts from the grammar's start symbol.
Bottom-Up parsing combines input words using SHIFT and
REDUCE operations.
CYK uses dynamic programming and a parsing table to
check whether a sentence follows a CFG in CNF.
============================================================
"""
