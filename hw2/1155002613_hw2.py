"""
File: 1155002613_hw2.py
Author: Mond Wan <1155002613>
Email: mondwan.1015@gmail.com
Github: 0
Description:
Train naive bayes classifier with data from business.txt and health.txt. Then,
classify data from test.txt and see how accurate it is.
"""


from __future__ import division
import os
# import sys
# from collections import defaultdict

import nltk
from nltk.probability import FreqDist
from nltk.text import TextCollection
from nltk.classify import NaiveBayesClassifier

# Initialize nltk library
REQUIRED_MODULES = ['punkt']
for m in REQUIRED_MODULES:
    nltk.download(m)

# features_dict = defaultdict(list)
data_table = {
    'health': {},
    'business': {},
}

for fn in [os.path.join('.', 'health.txt'), os.path.join('.', 'business.txt')]:
    # Get 'health' or 'business' currently
    current_label = os.path.splitext(os.path.basename(fn))[0]

    # Pre-process the trainning data
    with open(fn, 'r') as f:
        raw_text = ''.join(f.readlines())
        # remove non-ascii code
        raw_text = ''.join([i if ord(i) < 128 else ' ' for i in raw_text])

    # Tokenization and normalization
    tokens = [
        word.lower() for word in nltk.tokenize.word_tokenize(raw_text)
        if word.isalpha()
    ]

    # Prepare the unigram model
    freqdist = FreqDist()
    for word in tokens:
        freqdist[word] += 1

    # Calculate # of tokens
    N = freqdist.N()

    # Save useful data to data_table
    data_table[current_label]['raw_text'] = raw_text
    data_table[current_label]['tokens'] = tokens
    data_table[current_label]['freqdist'] = freqdist
    data_table[current_label]['N'] = N
    data_table[current_label]['U'] = len(freqdist.keys())

text_collection = TextCollection(
    [data_table[label]['tokens'] for label in data_table]
)


def extractor(word):
    """
    Capture features from the given word

    @param word string
    @return dict
      featuresets
    """
    featuresets = {}

    # Include Part of speech as a feature
    #
    # After running the program, we found that the POS of most of words in
    # business area is 'JJR' (Adjective, comparative).
    #
    # The result is reasonable since it is common to use wordings like
    # ```better``` or ```higher``` to do comparisons in business area
    featuresets['pos'] = nltk.pos_tag([word])[0][1]

    # Include last N characters as a feature
    #
    # Besides the part of speech, below features try to capture the suffix for
    # given english word.
    #
    # In fact, the # of N come from a try and error process. For N = 2 or 3,
    # different between health area and business area is very obvious
    featuresets['last-two-char'] = word[-2:]
    featuresets['last-three-char'] = word[-3:]

    for label in data_table:
        freqdist = data_table[label]['freqdist']
        N = data_table[label]['N']

        # tf = (freqdist[word] + 1) / (N + U)
        tf = round((freqdist[word]) / N, 3)
        idf = round(text_collection.idf(word), 3)

        # featuresets[label, 'unigram'] = tf

        # Include tfidf for a given label
        #
        # This piece of information tell us the importance of a word to a
        # document
        featuresets[label, 'tfidf'] = round(tf * idf, 3)

    # print word, featuresets
    return featuresets

# Extract features from the training data
features = [
    (extractor(word), label)
    for label in data_table for word in data_table[label]['tokens']
]

# Train the classifier
classifier = NaiveBayesClassifier.train(features)

with open(os.path.join('.', 'test.txt'), 'r') as f:
    raw_text = ''.join(f.readlines())
    # remove non-ascii code
    raw_text = ''.join([i if ord(i) < 128 else ' ' for i in raw_text])

tokens = [
    word.lower() for word in nltk.tokenize.word_tokenize(raw_text)
    # if word.isalpha()
]

results = [{
    'word': word,
    'category': classifier.classify(extractor(word))
} for word in tokens]

results = sorted(results, key=lambda s: s['word'])
with open(os.path.join('.', 'output.txt'), 'w') as f:
    for r in results:
        f.write('%s: %s\n' % (r['word'], r['category']))

classifier.show_most_informative_features()
