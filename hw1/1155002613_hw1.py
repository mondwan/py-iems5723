#!/usr/bin/python
"""
File: main.py
Author: Mond Wan (1155002613)
Email: mondwan.1015@gmail.com
Description: Homework1 main application
"""


import os
import sys

import nltk
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
# from nltk.probability import FreqDist

# Figure out the path for the input file
try:
    INPUT_FEED = sys.argv[1]
except IndexError:
    INPUT_FEED = os.path.join('.', 'input.txt')

# Read the input file
with open(INPUT_FEED, 'r') as f:
    INPUT_LINES = ''.join(f.readlines())

# Initialize nltk library
REQUIRED_MODULES = ['punkt']
for m in REQUIRED_MODULES:
    nltk.download(m)

# Parse and update data from input file as stated in the assignments
tokens = [
    word.lower() for word in nltk.tokenize.word_tokenize(INPUT_LINES)
    if word not in [',', '.']
]

# Bigram collocation
bigram_scores = []
unigram_scores = []
bc = BigramCollocationFinder.from_words(tokens)
bc_freq = bc.ngram_fd
uc_freq = bc.word_fd
N = uc_freq.N()

# Calculate the probability for unigram
for gram in uc_freq.items():
    val = float(uc_freq[gram[0]]) / N
    unigram_scores.append({
        'words': gram[0],
        'score': round(val, 2)
    })

# Calculate the probability for bigram
for gram in ngrams(tokens, 2):
    w1 = gram[0]
    w2 = gram[1]
    n_ii = bc_freq[(w1, w2)] / (bc.window_size - 1.0)
    n_ix = uc_freq[w1]
    bigram_scores.append({
        'words': '%s %s' % (w1, w2),
        'score': round(
            n_ii / n_ix,
            2
        ),
    })

# Sort by alphabetical and scores and order in descending order
unigram_scores = sorted(
    sorted(unigram_scores, key=lambda s: s['words']),
    key=lambda s: s['score'],
    reverse=True
)

bigram_scores = sorted(
    sorted(bigram_scores, key=lambda s: s['words']),
    key=lambda s: s['score'],
    reverse=True
)

lines = []
lines.append('Uni-gram:')
for score in unigram_scores[0:5]:
    lines.append('%.2f: %s' % (score['score'], score['words']))

lines.append('')

lines.append('Bi-gram:')
for score in bigram_scores[0:5]:
    lines.append('%.2f: %s' % (score['score'], score['words']))

with open(os.path.join('.', 'output.txt'), 'w') as f:
    f.write('\n'.join(lines))
    f.write('\n')
