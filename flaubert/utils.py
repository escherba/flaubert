import pandas as pd
from nltk.stem import wordnet


TREEBANK2WORDNET = {
    'J': wordnet.wordnet.ADJ,
    'V': wordnet.wordnet.VERB,
    'N': wordnet.wordnet.NOUN,
    'R': wordnet.wordnet.ADV
}


def read_tsv(file_input, iterator=False, chunksize=None):
    return pd.read_csv(
        file_input, iterator=iterator, chunksize=chunksize,
        header=0, quoting=2, delimiter="\t", escapechar="\\", quotechar='"',
        encoding="utf-8")


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
