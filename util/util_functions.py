import nltk
import numpy as np
import pickle as pkl
import itertools
from collections import Counter, OrderedDict
from keras.preprocessing.sequence import pad_sequences

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
    Input: list of all samples in the training data
    Return: OrderedDict - vocabulary mapping from word to integer.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*corpus))
    # Mapping from index to word (type: list)
    vocabulary = ['PADDING', 'UNKNOWN']   # 0 for padding, 1 for unknown words
    vocabulary = vocabulary.append([x[0] for x in word_counts.most_common()])
    # Mapping from word to index
    vocab2int = OrderedDict({x: i for i, x in enumerate(vocabulary)})
    return vocab2int

def pad_sent(sentences, max_words, max_sents):
	"""
	Pads sequences to the same length.
	Input: sentences - List of lists, where each element is a sequence.
					 - max_words: Int, maximum length of all sequences.
	"""
	
	# pad sentences in a doc
	sents_padded = pad_sequences(sentences, maxlen=max_words, padding='post') 
	# pad a doc to have equal number of sentences
	if len(sents_padded) < max_sents:
		doc_padding = np.zeros((max_sents-len(sents_padded),max_words), dtype = int)
		sents_padded = np.append(doc_padding, sents_padded, axis=0)
	else:
		sents_padded = sents_padded[:max_sents]

	return sents_padded



def build_input_data(corpus, vocab2int, max_words, max_sents):
    """
    Maps words in the corpus to integers based on a vocabulary.
    Also pad the sentences and documents into fixed shape
    Input: corpus - list of samples, each sample is a list of sentences, each sentence is a list of words
    """
    corpus_int = [[[vocab2int[word] for word in sentence]for sentence in sample] for sample in corpus]

    corpus_padded = []
    for doc in corpus_int:
    	corpus_padded.append(pad_sent(doc, max_words, max_sents))
    corpus_padded = np.array(corpus_padded)
     
    return corpus_padded













