# -*- coding: utf-8 -*-
'''
    These preprocessing utils would greatly benefit
    from a fast Cython rewrite.
'''
from __future__ import absolute_import

import string
import sys
import numpy as np
from collections import Counter

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f


def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]


def one_hot(text, n, filters=base_filter(), lower=True, split=" "):
    seq = text_to_word_sequence(text)
    return [(abs(hash(w)) % (n - 1) + 1) for w in seq]


class Tokenizer(object):
    def __init__(self, nb_words=None, filters=base_filter(), lower=True, split=" "):
        self.word_counts = Counter()
        self.word_docs = Counter()
        self.index_docs = Counter()
        self.document_count = 0
        self.filters = filters
        self.split = split
        self.lower = lower
        self.nb_words = nb_words

    def fit_on_texts(self, texts):
        '''
            required before using texts_to_sequences or texts_to_matrix
            @param texts: can be a list or a generator (for memory-efficiency)
        '''
        word_counts = Counter()
        word_docs = Counter()
        document_count = 0
        for text in texts:
            document_count += 1
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            word_counts.update(seq)
            word_docs.update(set(seq))
        self.document_count = document_count
        word_index = {w: i for i, (w, c) in enumerate(word_counts.most_common(), start=1)}
        self.index_docs = Counter({word_index[w]: c for w, c in word_docs.iteritems()})
        self.word_index = word_index

    def fit_on_sequences(self, sequences):
        '''
            required before using sequences_to_matrix
            (if fit_on_texts was never called)
        '''
        index_docs = Counter()
        document_count = 0
        for seq in sequences:
            document_count += 1
            index_docs.update(set(seq))
        self.document_count = document_count
        self.index_docs = index_docs

    def texts_to_sequences(self, texts):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Returns a list of sequences.
        '''
        res = []
        for vect in self.texts_to_sequences_generator(texts):
            res.append(vect)
        return res

    def texts_to_sequences_generator(self, texts):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Yields individual sequences.
        '''
        nb_words = self.nb_words
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        pass
                    else:
                        vect.append(i)
            yield vect

    def texts_to_matrix(self, texts, mode="binary"):
        '''
            modes: binary, count, tfidf, freq
        '''
        sequences = self.texts_to_sequences(texts)
        return self.sequences_to_matrix(sequences, mode=mode)

    def sequences_to_matrix(self, sequences, mode="binary"):
        '''
            modes: binary, count, tfidf, freq
        '''
        if not self.nb_words:
            if self.word_index:
                nb_words = len(self.word_index)
            else:
                raise Exception("Specify a dimension (nb_words argument), or fit on some text data first")
        else:
            nb_words = self.nb_words

        if mode == "tfidf" and not self.document_count:
            raise Exception("Fit the Tokenizer on some data before using tfidf mode")

        X = np.zeros((len(sequences), nb_words))
        index_docs = self.index_docs
        document_count1 = float(self.document_count + 1)
        for i, seq in enumerate(sequences):
            if not seq:
                pass
            seq_len = len(seq)
            counts = Counter()
            for j in seq:
                if j < nb_words:
                    counts[j] += 1
            for j, c in counts.iteritems():
                if mode == "count":
                    X[i][j] = float(c)
                elif mode == "freq":
                    X[i][j] = float(c) / seq_len
                elif mode == "binary":
                    X[i][j] = 1
                elif mode == "tfidf":
                    tf = np.log(float(c) / seq_len)
                    df = (1.0 + np.log(1.0 + index_docs.get(j, 0) / document_count1))
                    X[i][j] = tf / df
                else:
                    raise Exception("Unknown vectorization mode: " + str(mode))
        return X
