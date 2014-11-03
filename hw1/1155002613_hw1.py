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
from nltk.probability import FreqDist

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
    word.lower() for word in nltk.tokenize.word_tokenize(INPUT_LINES) \
    if word not in [',', '.']
]

# Bigram collocation
bigram_scores = []
unigram_scores = []
bc = BigramCollocationFinder.from_words(tokens)
uc = bc.word_fd

# Calculate the probability for unigram
for gram in uc.items():
    unigram_scores.append({
        'words': gram[0],
        'score': uc.freq(gram[0])
    })

# Calculate the probability for bigram
for gram in ngrams(tokens, 2):
    w1 = gram[0]
    w2 = gram[1]
    bigram_scores.append({
        'words': '%s %s' % (w1, w2),
        'score': bc.score_ngram(
            lambda c, b, n: c / b[0],
            w1,
            w2
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

print 'Uni-gram:'
for score in unigram_scores[0:5]:
    print '%.2f: %s' % (score['score'], score['words'])

print ''

print 'Bi-gram:'
for score in bigram_scores[0:5]:
    print '%.2f: %s' % (score['score'], score['words'])
