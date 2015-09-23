import pandas
import random
import numpy as np
import regex as re
from itertools import chain
from collections import defaultdict, Counter
from nltk.stem import wordnet
from scipy.sparse import dok_matrix as sparse_matrix_type
from fastcache import clru_cache
from sklearn.base import BaseEstimator, TransformerMixin
from pymaptools.vectorize import enumerator
from pymaptools.iter import isiterable
from pymaptools.io import read_json_lines
from flaubert.conf import CONFIG


TREEBANK2WORDNET = {
    'J': wordnet.wordnet.ADJ,
    'V': wordnet.wordnet.VERB,
    'N': wordnet.wordnet.NOUN,
    'R': wordnet.wordnet.ADV
}


def json_field_iter(files, field=None):
    for fname in files:
        for doc in read_json_lines(fname):
            yield doc if field is None else doc[field]


def read_tsv(file_input, iterator=False, chunksize=None):
    """
    @rtype: pandas.core.frame.DataFrame
    """
    return pandas.read_csv(
        file_input, iterator=iterator, chunksize=chunksize,
        **CONFIG['data']['read_csv_kwargs'])


def pd_row_iter(datasets, chunksize=1000, field=None):
    """Produce an iterator over rows or field values in Pandas
    dataframe while reading from files on disk

    @param datasets: a list of filenames or file handles
    @param chunksize: how many lines to read at once
    """
    # ensure that silly values of chunksize don't get passed
    if not chunksize:
        chunksize = 1
    if not isiterable(datasets):
        datasets = [datasets]
    for dataset in datasets:
        for chunk in read_tsv(dataset, iterator=True, chunksize=chunksize):
            iterable = chunk.iterrows() \
                if field is None \
                else chunk[field]
            for val in iterable:
                yield val


def pd_dict_iter(datasets, chunksize=1000):
    for idx, row in pd_row_iter(datasets, chunksize=chunksize):
        yield dict(row)


def reservoir_list(iterator, K, random_state=0):
    """Simple reservoir sampler
    """
    random.seed(random_state)
    sample = []
    for idx, item in enumerate(iterator):
        if len(sample) < K:
            sample.append(item)
        else:
            # accept with probability K / idx
            sample_idx = int(random.random() * idx)
            if sample_idx < K:
                sample[sample_idx] = item
    return sample


def reservoir_dict(iterator, field, Kdict, random_state=0):
    """Reservoir sampling over a list of dicts

    Given a field, and a mapping of field values to integers K,
    return a sample from the iterator such that for the field specified,
    each value occurs at most K times.  For example, for a binary ouput
    value Y, we would request

        field='Y', Kdict={0: 500, 1: 1000}

    to return 500 instances of Y=0 and 1000 instances of Y=1
    """
    random.seed(random_state)
    sample = defaultdict(list)
    field_indices = Counter()
    for row in iterator:
        field_val = row[field]
        if field_val in Kdict:
            idx = field_indices[field_val]
            field_list = sample[field_val]
            if len(field_list) < Kdict[field_val]:
                field_list.append(row)
            else:
                # accept with probability K / idx
                sample_idx = int(random.random() * idx)
                if sample_idx < Kdict[field_val]:
                    field_list[sample_idx] = row
            field_indices[field_val] += 1
    return list(chain.from_iterable(sample.itervalues()))


def lru_wrap(func, cache_size=None):
    if cache_size:
        return clru_cache(maxsize=cache_size)(func)
    else:
        return func


def treebank2wordnet(treebank_tag):
    if not treebank_tag:
        return None
    letter = treebank_tag[0]
    return TREEBANK2WORDNET.get(letter)


def sum_dicts(*args):
    """Return a sum total of several dictionaries
    Note: this is non-commutative (later entries override earlier)"""
    result = dict()
    for arg in args:
        result.update(arg)
    return result


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.
    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.
    >> len(data[key]) == n_samples
    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).
    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.
    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)
    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.
    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


# TODO: make BagVectorizer support scikit-learn pipelines

class BagVectorizer(BaseEstimator, TransformerMixin):

    """Transform an array of word bags into a sparse matrix

    Similar to DictVectorizer except taks list of lists instead
    of list of dicts and can be pre-initialized with a vocabulary
    """
    def __init__(self, vocabulary=None, sparse=True, onehot=True,
                 stop_words=None, dtype=np.int64):
        self.vocabulary_ = vocabulary
        self.sparse_ = sparse
        self.dtype_ = dtype
        self.onehot_ = onehot
        if isinstance(stop_words, basestring):
            stop_words = re.findall(u'\\w+', stop_words)
        elif stop_words is None:
            stop_words = []
        self.stop_words_ = frozenset(stop_words)

    def fit(self, X, y=None):
        enum = self.vocabulary_ or enumerator()
        for row in X:
            for lbl in row:
                enum[lbl]
        self.vocabulary_ = enum
        self.feature_names_ = enum.keys()
        return self

    def transform(self, X, y=None):

        enum = self.vocabulary_
        shape = (len(X), len(self.vocabulary_))
        mat = sparse_matrix_type(shape, dtype=self.dtype_)
        stop_words = self.stop_words_
        if self.onehot_:
            for idx, row in enumerate(X):
                for word in row:
                    if word not in stop_words:
                        mat[idx, enum[word]] = 1
        else:
            for idx, row in enumerate(X):
                for word, count in Counter(row).iteritems():
                    if word not in stop_words:
                        mat[idx, enum[word]] = count
        self.feature_names_ = enum.keys()
        return mat.tocsr() if self.sparse_ else mat.todense()

    def get_feature_names(self):
        return self.feature_names_


def drop_nans(X, y):
    """Drop rows which contain NaNs

    Assumes X and y are of equal length, but X is a matrix
    and y is a vector
    """
    X_nans = np.isnan(X).any(axis=1)
    y_nans = np.asarray(np.isnan(y))
    nans = X_nans | y_nans
    y = y[~nans]
    X = X[~nans]
    return X, y
