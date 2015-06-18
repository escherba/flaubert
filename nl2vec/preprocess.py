import nltk
import pandas as pd
import unicodedata
import regex as re
import sys
import abc
import warnings
from itertools import islice
from bs4 import BeautifulSoup
from functools import partial
from nltk.corpus import stopwords
from nltk.stem import wordnet, PorterStemmer
from nltk import pos_tag
from joblib import Parallel, delayed
from fastcache import clru_cache
from pymaptools.io import write_json_line, PathArgumentParser, GzipFileType
from nl2vec.conf import CONFIG


TREEBANK2WORDNET = {
    'J': wordnet.wordnet.ADJ,
    'V': wordnet.wordnet.VERB,
    'N': wordnet.wordnet.NOUN,
    'R': wordnet.wordnet.ADV
}


def treebank2wordnet(treebank_tag):
    if not treebank_tag:
        return None
    letter = treebank_tag[0]
    return TREEBANK2WORDNET.get(letter)


class Replacer(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def replace(self, text):
        """Calls regex's sub method with self as callable"""

    @abc.abstractmethod
    def replacen(self, text):
        """Calls regex's subn method with self as callable"""


class RepeatReplacer(Replacer):
    """Remove repeating characters from text

    The default pattern only applies to non-decimal characters

    >>> rep = RepeatReplacer(max_repeats=3)
    >>> rep.replace(u"So many $100000 bills.......")
    u'So many $100000 bills...'
    """
    def __init__(self, pattern=u'[^\\d]', max_repeats=3):
        if max_repeats < 1:
            raise ValueError("Invalid parameter value max_repeats={}"
                             .format(max_repeats))
        prev = u'\\1' * max_repeats
        pattern = u'(%s)' % pattern
        regexp = re.compile(pattern + prev + u'+', re.UNICODE)
        self.replace = partial(regexp.sub, prev)
        self.replacen = partial(regexp.subn, prev)

    def replace(self):
        """Remove repeating characters from text

        Method definition needed only for abstract base class
        (it is overwritten during init)
        """
        pass

    def replacen(self):
        """Remove repeating characters from text
        while also returning number of substitutions made

        Method definition needed only for abstract base class
        (it is overwritten during init)
        """
        pass


class SimpleSentenceTokenizer(object):

    def __init__(self, lemmatizer=None, stemmer=None,
                 word_regex=u"\\p{L}+", form='NFKC', stop_words="english",
                 max_char_repeats=3, lru_cache_size=50000):
        self._unicode_normalize = partial(unicodedata.normalize, form)
        self._tokenize = re.compile(word_regex, re.UNICODE | re.IGNORECASE).findall
        self._stopwords = frozenset(stopwords.words(stop_words))
        self._sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self._lemmatize = clru_cache(maxsize=lru_cache_size)(lemmatizer.lemmatize) if lemmatizer else None
        self._stem = stemmer.stem if stemmer else None
        self._pos_tag = pos_tag
        self._replace_char_repeats = RepeatReplacer(max_repeats=max_char_repeats).replace \
            if max_char_repeats > 0 \
            else lambda x: x

    def strip_html(self, text):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = BeautifulSoup(text).get_text()
        return text

    def word_tokenize(self, text, remove_html=True, remove_stopwords=True):
        # 1. Remove HTML
        if remove_html:
            text = self.strip_html(text)

        # 2. Normalize Unicode
        text = self._unicode_normalize(text)

        # 3. Lowercase
        text = text.lower()

        # 4. Reduce repeated characters to specified number (usually 3)
        text = self._replace_char_repeats(text)

        # 5. Tokenize on letters only (simple)
        words = self._tokenize(text)

        # 6. Lemmatize or stem based on POS tags
        if self._lemmatize:
            final_words = []
            lemmatize = self._lemmatize
            for word, tag in self._pos_tag(words):
                wordnet_tag = treebank2wordnet(tag)
                if wordnet_tag is not None:
                    word = lemmatize(word, pos=wordnet_tag)
                final_words.append(word)
            words = final_words
        elif self._stem:
            stem = self._stem
            words = [stem(word) for word in words]

        # 7. Optionally remove stop words (false by default)
        if remove_stopwords:
            stop_words = self._stopwords
            words = [word for word in words if word not in stop_words]

        # 8. Return a list of words
        return words

    def sentence_tokenize(self, text, remove_html=True, remove_stopwords=False):
        if remove_html:
            text = self.strip_html(text)
        sentences = []
        for raw_sentence in self._sentence_tokenizer.tokenize(text):
            if not raw_sentence:
                continue
            words = self.word_tokenize(raw_sentence, remove_html=remove_html,
                                       remove_stopwords=remove_stopwords)
            if not words:
                continue
            sentences.append(words)
        return sentences

    tokenize = word_tokenize


def read_tsv(file_input, iterator=False, chunksize=None):
    return pd.read_csv(
        file_input, iterator=iterator, chunksize=chunksize,
        header=0, quoting=2, delimiter="\t", escapechar="\\", quotechar='"',
        encoding="utf-8")


def get_field_iter(field, datasets, chunksize=1000):
    """Produce an iterator over values for a particular field in Pandas
    dataframe while reading from disk

    :param field: a string specifying field name of interest
    :param datasets: a list of filenames or file handles
    :param chunksize: how many lines to read at once
    """
    # ensure that silly values of chunksize don't get passed
    if not chunksize:
        chunksize = 1
    for dataset in datasets:
        for chunk in read_tsv(dataset, iterator=True, chunksize=chunksize):
            for review in chunk[field]:
                yield review


REGISTRY = {
    None: None,
    'wordnet': wordnet.WordNetLemmatizer(),
    'porter': PorterStemmer()
}


TOKENIZER = SimpleSentenceTokenizer(
    lemmatizer=REGISTRY[CONFIG['lemmatizer']],
    stemmer=REGISTRY[CONFIG['stemmer']],
    **CONFIG['tokenizer'])


def get_sentences(text, **kwargs):
    sentences = []
    for sentence in TOKENIZER.sentence_tokenize(text, **kwargs):
        sentences.append(sentence)
    return sentences


def get_words(review, **kwargs):
    return TOKENIZER.tokenize(review, **kwargs)


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--input', type=GzipFileType('r'), default=[sys.stdin], nargs='*',
                        help='Input file (in TSV format, optionally compressed)')
    parser.add_argument('--field', type=str, default='review',
                        help='Field name (Default: review)')
    parser.add_argument('--sentences', action='store_true',
                        help='split by sentence instead of by record')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process this many lines (for testing)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help="Number of jobs to run")
    parser.add_argument('--output', type=GzipFileType('w'), default=sys.stdout,
                        help='File to write sentences to, optionally compressed')
    namespace = parser.parse_args(args)
    return namespace


def get_review_iterator(args):
    iterator = get_field_iter(args.field, args.input, chunksize=1000)
    if args.limit:
        iterator = islice(iterator, args.limit)
    return iterator


def get_tokenize_method(args):
    if args.sentences:
        tokenize = get_sentences
    else:
        tokenize = get_words
    return tokenize


def run(args):
    iterator = get_review_iterator(args)
    tokenize = get_tokenize_method(args)
    write_record = partial(write_json_line, args.output)
    if args.n_jobs == 1:
        # turn off parallelism
        for review in iterator:
            record = tokenize(review)
            write_record(record)
    else:
        # enable parallellism
        for record in Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(tokenize)(review) for review in iterator):
            write_record(record)


if __name__ == "__main__":
    run(parse_args())
