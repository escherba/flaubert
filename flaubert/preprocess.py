import nltk
import unicodedata
import regex as re
import sys
import abc
import logging
import os
import cPickle as pickle
from pkg_resources import resource_filename
from bs4 import BeautifulSoup
from itertools import islice
from functools import partial
from nltk.corpus import stopwords
from nltk.stem import wordnet, PorterStemmer
from nltk import pos_tag
from joblib import Parallel, delayed
from fastcache import clru_cache
from pymaptools.io import write_json_line, PathArgumentParser, GzipFileType, open_gz
from flaubert.tokenize import RegexpFeatureTokenizer
from flaubert.urls import URLParser
from flaubert.conf import CONFIG
from flaubert.HTMLParser import HTMLParser, HTMLParseError
from flaubert.utils import read_tsv, treebank2wordnet
from flaubert.unicode_maps import EXTRA_TRANSLATE_MAP


logging.basicConfig(level=logging.WARN)
LOG = logging.getLogger(__name__)


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


class GenericReplacer(Replacer):

    __metaclass__ = abc.ABCMeta

    def __init__(self, re):
        self._re = re

    @abc.abstractmethod
    def __call__(self, match):
        """Override this to provide your own substitution method"""

    def replace(self, text):
        return self._re.sub(self, text)

    def replacen(self, text):
        return self._re.subn(self, text)


class InPlaceReplacer(GenericReplacer):

    def __init__(self, replace_map=None):
        if replace_map is None:
            replace_map = dict()
        _replacements = dict()
        _regexes = list()
        for idx, (key, val) in enumerate(replace_map.iteritems()):
            _replacements[idx] = val
            _regexes.append(u'({})'.format(key))
        self._replacements = _replacements
        super(InPlaceReplacer, self).__init__(
            re=re.compile(u'|'.join(_regexes), re.UNICODE | re.IGNORECASE))

    def __call__(self, match):
        lastindex = match.lastindex
        if lastindex is None:
            return u''
        replacement = self._replacements[lastindex - 1]
        matched_string = match.group(lastindex)
        return replacement.get(matched_string.lower(), matched_string) \
            if isinstance(replacement, dict) \
            else replacement


class Translator(Replacer):
    """Replace certain characters
    """
    def __init__(self, translate_mapping=None, translated=False):
        if translated:
            self._translate_map = dict((translate_mapping or {}).iteritems())
        else:
            self._translate_map = {ord(k): ord(v) for k, v in (translate_mapping or {}).iteritems()}

    def add_inverse_map(self, inverse_mapping, translated=False):
        replace_map = {}
        for key, vals in (inverse_mapping or {}).iteritems():
            for val in vals:
                replace_map[val] = key
        self.add_map(replace_map, translated=translated)

    def add_map(self, mapping, translated=False):
        replace_map = self._translate_map
        if translated:
            replace_map.update(mapping)
        else:
            for key, val in mapping.iteritems():
                replace_map[ord(key)] = ord(val)

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
                 sentence_tokenizer=('nltk_data', 'tokenizers/punkt/english.pickle'),
                 max_char_repeats=3, lru_cache_size=50000, translate_map_inv=None,
                 replace_map=None, html_renderer='default', add_abbrev_types=None,
                 del_sent_starters=None):
        self._unicode_normalize = partial(unicodedata.normalize, unicode_form)
        self._replace_inplace = InPlaceReplacer(replace_map).replace \
            if replace_map else lambda x: x
        self._tokenize = RegexpFeatureTokenizer().tokenize
        self._stopwords = frozenset(stopwords.words(nltk_stop_words))
        self._url_parser = url_parser

        self._sentence_tokenizer = None
        if sentence_tokenizer is None:
            self._sentence_tokenize = lambda x: [x]
        elif sentence_tokenizer[0] == 'nltk_data':
            punkt = nltk.data.load(sentence_tokenizer[1])
            self._sentence_tokenize = punkt.tokenize
        elif sentence_tokenizer[0] == 'data':
            tokenizer_path = os.path.join('..', 'data', sentence_tokenizer[1])
            tokenizer_path = resource_filename(__name__, tokenizer_path)
            if os.path.exists(tokenizer_path):
                with open_gz(tokenizer_path, 'rb') as fhandle:
                    punkt = pickle.load(fhandle)
                if add_abbrev_types:
                    punkt._params.abbrev_types = punkt._params.abbrev_types | set(add_abbrev_types)
                if del_sent_starters:
                    punkt._params.sent_starters = punkt._params.sent_starters - set(del_sent_starters)
                self._sentence_tokenize = punkt.sentences_from_text
                self._sentence_tokenizer = punkt
            else:
                self._sentence_tokenize = lambda x: [x]
                logging.warn("Tokenizer not found at %s" % tokenizer_path)
        else:
            raise ValueError("Invalid sentence tokenizer class")

        self.sentence_tokenizer = None
        self._lemmatize = clru_cache(maxsize=lru_cache_size)(lemmatizer.lemmatize) if lemmatizer else None
        self._stem = stemmer.stem if stemmer else None
        self._pos_tag = pos_tag
        self._replace_char_repeats = \
            RepeatReplacer(max_repeats=max_char_repeats).replace \
            if max_char_repeats > 0 else self._identity

        # translation of Unicode characters
        translator = Translator(EXTRA_TRANSLATE_MAP, translated=True)
        translator.add_inverse_map(translate_map_inv, translated=False)
        self._replace_chars = translator.replace

        if html_renderer is None:
            self.strip_html = lambda x: x
        elif html_renderer == u'default':
            self.strip_html = HTMLCleaner().clean
        elif html_renderer == u'beautifulsoup':
            self.strip_html = self._strip_html_bs
        else:
            raise ValueError('Invalid parameter value given for `html_renderer`')

        # tokenize a dummy string b/c lemmatizer and/or other tools can take
        # a while to initialize screwing up our attempts to measure performance
        self.tokenize(u"dummy string")

    @staticmethod
    def _identity(arg):
        return arg

    def unicode_normalize(self, text):
        # 1. Normalize to specific Unicode form (also replaces ellipsis with
        # periods)
        text = self._unicode_normalize(text)
        # 2. Replace certain chars such as n- and m-dashes
        text = self._replace_inplace(text)
        return text

    def preprocess(self, text, lowercase=True):
        # 1. Remove HTML
        text = self.strip_html(text)
        # 2. Normalize Unicode
        text = self.unicode_normalize(text)
        # 3. Replace certain characters
        text = self._replace_chars(text)
        # 4. whiteout URLs
        text = self._url_parser.whiteout_urls(text)
        # 5. Lowercase
        if lowercase:
            text = text.lower()
        # 6. Reduce repeated characters to specified number (usually 3)
        text = self._replace_char_repeats(text)
        return text

    def _strip_html_bs(self, text):
        """
        Use BeautifulSoup to strip off HTML but in such a way that <BR> and
        <P> tags get rendered as new lines
        """
        soup = BeautifulSoup(text)
        fragments = []
        for element in soup.recursiveChildGenerator():
            if isinstance(element, basestring):
                fragments.append(element.strip())
            elif element.name == 'br':
                fragments.append(u"\n")
            elif element.name == 'p':
                fragments.append(u"\n")
        result = u"".join(fragments).strip()
        return result

    def word_tokenize(self, text, lowercase=True, preprocess=True, remove_stopwords=False):
        # 1. Misc. preprocessing
        if preprocess:
            text = self.preprocess(text, lowercase=lowercase)
        elif lowercase:
            text = text.lower()

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
            text = self.preprocess(text, lowercase=False)

        sentences = []
        for raw_sentence in self._sentence_tokenize(text):
            if not raw_sentence:
                continue
            words = self.word_tokenize(
                raw_sentence, preprocess=False, lowercase=True,
                remove_stopwords=remove_stopwords)
            if not words:
                continue
            sentences.append(words)
        return sentences

    tokenize = word_tokenize


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


def registry(key):
    if key is None:
        return None
    elif key == 'wordnet':
        return wordnet.WordNetLemmatizer()
    elif key == 'porter':
        return PorterStemmer()


def tokenizer_builder():
    return SimpleSentenceTokenizer(
        lemmatizer=registry(CONFIG[__name__]['lemmatizer']),
        stemmer=registry(CONFIG[__name__]['stemmer']),
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


def get_review_iterator(args):
    iterator = get_field_iter(args.field, args.input, chunksize=1000)
    if args.limit:
        iterator = islice(iterator, args.limit)
    return iterator


def get_mapper_method(args):
    if args.sentences:
        mapper = get_sentences
    else:
        mapper = get_words
    return mapper


def run_tokenize(args):
    iterator = get_review_iterator(args)
    mapper = get_mapper_method(args)
    write_record = partial(write_json_line, args.output)
    if args.n_jobs == 1:
        # turn off parallelism
        for review in iterator:
            record = mapper(review)
            write_record(record)
    else:
        # enable parallellism
        for record in Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(mapper)(review) for review in iterator):
            write_record(record)


def run_train(args):
    from flaubert.punkt import PunktTrainer, \
        PunktLanguageVars, PunktSentenceTokenizer
    iterator = get_review_iterator(args)
    reviews = []
    for review in iterator:
        review = TOKENIZER.preprocess(review, lowercase=False).strip()
        if not review.endswith(u'.'):
            review += u'.'
        reviews.append(review)
    text = u'\n\n'.join(reviews)
    custom_lang_vars = PunktLanguageVars
    custom_lang_vars.sent_end_chars = ('.', '?', '!')

    # TODO: check if we need to manually specify common abbreviations
    punkt = PunktTrainer(verbose=args.verbose, lang_vars=custom_lang_vars())
    abbrev_sent = u'Start %s end.' % u' '.join(CONFIG['tokenizer']['add_abbrev_types'])
    punkt.train(abbrev_sent, finalize=False)
    punkt.train(text, finalize=False)
    punkt.finalize_training()
    model = PunktSentenceTokenizer(punkt.get_params())
    pickle.dump(model, args.output, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--input', type=GzipFileType('r'), default=[sys.stdin], nargs='*',
                        help='Input file (in TSV format, optionally compressed)')
    parser.add_argument('--field', type=str, default='review',
                        help='Field name (Default: review)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process this many lines (for testing)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help="Number of jobs to run")
    parser.add_argument('--output', type=GzipFileType('w'), default=sys.stdout,
                        help='Output file')

    subparsers = parser.add_subparsers()

    parser_tokenize = subparsers.add_parser('tokenize')
    parser_tokenize.add_argument('--sentences', action='store_true',
                                 help='split on sentences')
    parser_tokenize.set_defaults(func=run_tokenize)

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--verbose', action='store_true',
                              help='be verbose')
    parser_train.set_defaults(func=run_train)

    namespace = parser.parse_args(args)
    return namespace


def run():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    run()
