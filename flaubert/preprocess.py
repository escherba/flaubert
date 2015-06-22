import nltk
import pandas as pd
import unicodedata
import regex as re
import sys
import abc
import logging
from itertools import islice
from functools import partial
from nltk.corpus import stopwords
from nltk.stem import wordnet, PorterStemmer
from nltk import pos_tag
from joblib import Parallel, delayed
from fastcache import clru_cache
from pymaptools.io import write_json_line, PathArgumentParser, GzipFileType
from flaubert.tokenize import RegexpFeatureTokenizer
from flaubert.urls import URLParser
from flaubert.conf import CONFIG
from flaubert.HTMLParser import HTMLParser, HTMLParseError


TREEBANK2WORDNET = {
    'J': wordnet.wordnet.ADJ,
    'V': wordnet.wordnet.VERB,
    'N': wordnet.wordnet.NOUN,
    'R': wordnet.wordnet.ADV
}


logging.basicConfig(level=logging.WARN)
LOG = logging.getLogger(__name__)


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
    def __init__(self, pattern=u'[^\\d\\*]', max_repeats=3):
        if max_repeats < 1:
            raise ValueError("Invalid parameter value max_repeats={}"
                             .format(max_repeats))
        prev = u'\\1' * max_repeats
        pattern = u'(%s)' % pattern
        regexp = re.compile(pattern + prev + u'+', re.UNICODE)
        self.replace = partial(regexp.sub, prev)
        self.replacen = partial(regexp.subn, prev)

    def replace(self, text):
        """Remove repeating characters from text

        Method definition needed only for abstract base class
        (it is overwritten during init)
        """
        pass

    def replacen(self, text):
        """Remove repeating characters from text
        while also returning number of substitutions made

        Method definition needed only for abstract base class
        (it is overwritten during init)
        """
        pass


class Translator(Replacer):
    """Replace certain characters
    """
    def __init__(self, translate_map):
        self._translate_map = {ord(k): ord(v) for k, v in translate_map.iteritems()}

    @classmethod
    def from_inverse_map(cls, inverse_map):
        replace_map = {}
        for key, vals in (inverse_map or {}).iteritems():
            for val in vals:
                replace_map[val] = key
        return cls(replace_map)

    def replace(self, text):
        """Replace characters
        """
        return text.translate(self._translate_map)

    def replacen(self, text):
        """Replace characters
        while also returning number of substitutions made

        Method definition needed only for abstract base class
        """
        pass


class MLStripper(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.reset()
        self.fed = []
        self.handled_starttags = []
        self.handled_startendtags = []
        self._new_lines = 0

    def append_new_lines(self):
        for _ in xrange(self._new_lines):
            self.fed.append("\n")
            self._new_lines = 0

    def handle_data(self, data):
        self.append_new_lines()
        self.fed.append(data)

    def handle_starttag(self, tag, attrs):
        HTMLParser.handle_starttag(self, tag, attrs)
        self.handled_starttags.append(tag)
        if tag == u"br":
            self._new_lines += 1
        elif tag == u"p":
            self._new_lines += 1

    def handle_endtag(self, tag):
        HTMLParser.handle_endtag(self, tag)
        if tag == u"p":
            self._new_lines += 1

    def handle_startendtag(self, tag, attrs):
        HTMLParser.handle_starttag(self, tag, attrs)
        self.handled_startendtags.append(tag)
        if tag == u"br":
            self._new_lines += 1

    def handle_entityref(self, name):
        # Ignore HTML entities (already unescaped)
        self.fed.append(u'&' + name)

    def get_data(self):
        self.append_new_lines()
        return u''.join(self.fed)


class HTMLCleaner(object):

    _remove_full_comment = partial(
        (re.compile(ur"(?s)<!--(.*?)-->[\n]?", re.UNICODE)).sub, ur'\1')
    _remove_partial_comment = partial(
        (re.compile(ur"<!--", re.UNICODE)).sub, u"")

    def __init__(self, strip_html=True, strip_html_comments=True):
        self._strip_html = strip_html
        self._strip_html_comments = strip_html_comments

    def clean(self, html):
        """Remove HTML markup from the given string
        """
        if self._strip_html_comments:
            html = self._remove_full_comment(html)
            html = self._remove_partial_comment(html)
        if html and self._strip_html:
            stripper = MLStripper()
            try:
                stripper.feed(html)
            except HTMLParseError as err:
                logging.exception(err)
            else:
                html = stripper.get_data().strip()
        return html


class SimpleSentenceTokenizer(object):

    def __init__(self, lemmatizer=None, stemmer=None, url_parser=None,
                 unicode_form='NFKC', nltk_stop_words="english",
                 nltk_sentence_tokenizer='tokenizers/punkt/english.pickle',
                 max_char_repeats=3, lru_cache_size=50000, replace_map=None):
        self._unicode_normalize = partial(unicodedata.normalize, unicode_form)
        self._tokenize = RegexpFeatureTokenizer().tokenize
        self._stopwords = frozenset(stopwords.words(nltk_stop_words))
        self._url_parser = url_parser
        self._sentence_tokenize = nltk.data.load(nltk_sentence_tokenizer).tokenize
        self._lemmatize = clru_cache(maxsize=lru_cache_size)(lemmatizer.lemmatize) if lemmatizer else None
        self._stem = stemmer.stem if stemmer else None
        self._pos_tag = pos_tag
        self._replace_char_repeats = \
            RepeatReplacer(max_repeats=max_char_repeats).replace \
            if max_char_repeats > 0 else self._identity
        self._replace_chars = Translator.from_inverse_map(replace_map).replace
        self.strip_html = HTMLCleaner().clean

        # tokenize a dummy string b/c lemmatizer and/or other tools can take
        # a while to initialize screwing up our attempts to measure performance
        self.tokenize(u"dummy string")

    @staticmethod
    def _identity(arg):
        return arg

    def _preprocess_text(self, text):
        # 1. Remove HTML
        text = self.strip_html(text)
        # 2. Normalize Unicode
        text = self._unicode_normalize(text)
        # 3. Replace certain characters
        text = self._replace_chars(text)
        # 4. whiteout URLs
        text = self._url_parser.whiteout_urls(text)
        # 5. Lowercase
        text = text.lower()
        # 6. Reduce repeated characters to specified number (usually 3)
        text = self._replace_char_repeats(text)
        return text

    def word_tokenize(self, text, preprocess=True, remove_stopwords=True):
        # 1. Misc. preprocessing
        if preprocess:
            text = self._preprocess_text(text)

        # 2. Tokenize
        words = self._tokenize(text)

        # 3. Lemmatize or stem based on POS tags
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

        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stop_words = self._stopwords
            words = [word for word in words if word not in stop_words]

        # 5. Return a list of words
        return words

    def sentence_tokenize(self, text, preprocess=True,
                          remove_stopwords=False):
        if preprocess:
            text = self._preprocess_text(text)

        sentences = []
        for raw_sentence in self._sentence_tokenize(text):
            if not raw_sentence:
                continue
            words = self.word_tokenize(
                raw_sentence, preprocess=False,
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


def tokenizer_builder():
    return SimpleSentenceTokenizer(
        lemmatizer=REGISTRY[CONFIG['lemmatizer']],
        stemmer=REGISTRY[CONFIG['stemmer']],
        url_parser=URLParser(),
        **CONFIG['tokenizer'])


TOKENIZER = tokenizer_builder()


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
