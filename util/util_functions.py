import nltk
import numpy as np
import pickle as pkl
import itertools
from collections import Counter

def sent_tokenize(doc):
	sent_text = nltk.sent_tokenize(doc) # this gives you a list of sentences
	return sent_text

def word_tokenize(sent):
	tokenized_text = nltk.word_tokenize(sent)  # this gives you a list of words
	return tokenized_text

def pos_tag(tokenized_text):
	# POS tagging. Input is tokenized text
	tagged = nltk.pos_tag(tokenized_text)
	return tagged


def build_vocab(corpus):
    """
    Builds a vocabulary mapping from word to index based on the corpus.
    Input: list of samples, each sample is a list of words
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*corpus))
    # Mapping from index to word (type: list)
    vocabulary_inv = ['PADDING', 'UNKNOWN']   # 0 for padding, 1 for unknown words
    vocabulary_inv = vocabulary_inv.append([x[0] for x in word_counts.most_common()])
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x