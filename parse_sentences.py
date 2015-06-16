import nltk
import pandas as pd
import unicodedata
import regex as re
import sys
from itertools import islice
from bs4 import BeautifulSoup
from functools import partial
from nltk.corpus import stopwords
from nltk.stem import wordnet, PorterStemmer
from nltk import pos_tag
from joblib import Parallel, delayed
from pymaptools.io import write_json_line, PathArgumentParser, GzipFileType


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


class SimpleSentenceTokenizer(object):

    def __init__(self, word_regex=u"\\p{L}+", form='NFKC', stop_words=stopwords.words("english"),
                 lemmatizer=None, stemmer=None):
        self._normalize = partial(unicodedata.normalize, form)
        self._word_tokenize = re.compile(word_regex, re.UNICODE | re.IGNORECASE).findall
        self._stopwords = frozenset(stop_words)
        self._sentence_tokenize = nltk.data.load('tokenizers/punkt/english.pickle').tokenize
        self._lemmatizer = lemmatizer
        self._stemmer = stemmer
        self._pos_tag = pos_tag

    def word_tokenize(self, text, remove_html=True, remove_stopwords=False):
        # 1. Remove HTML
        if remove_html:
            text = BeautifulSoup(text).get_text()

        # 2. Normalize Unicode
        text = self._normalize(text)

        # 3. Lowercase
        text = text.lower()

        # 4. Tokenize on letters only (simple)
        words = self._word_tokenize(text)

        # 5. Lemmatize or stem based on POS tags
        if self._lemmatizer:
            words = []
            lemmatize = self._lemmatizer.lemmatize
            for word, tag in self._pos_tag(words):
                wordnet_tag = treebank2wordnet(tag)
                if wordnet_tag is not None:
                    word = lemmatize(word, pos=wordnet_tag)
                words.append(word)
        elif self._stemmer:
            stem = self._stemmer.stem
            words = [stem(word) for word in words]

        # 6. Optionally remove stop words (false by default)
        if remove_stopwords:
            stop_words = self._stopwords
            words = [word for word in words if word not in stop_words]

        # 7. Return a list of words
        return words

    def sentence_tokenize(self, text):
        text = BeautifulSoup(text).get_text()
        sentences = []
        for raw_sentence in self._sentence_tokenize(text):
            if not raw_sentence:
                continue
            words = self.word_tokenize(raw_sentence, remove_html=False)
            if not words:
                continue
            sentences.append(words)
        return sentences

    tokenize = word_tokenize


def read_tsv(file_input):
    return pd.read_csv(file_input, header=0, delimiter="\t", quoting=2,
                       escapechar="\\", quotechar='"', encoding="utf-8")


def get_reviews(*datasets):
    for fhandle in datasets:
        dataset = read_tsv(fhandle)
        for review in dataset['review']:
            yield review


def get_sentences(tokenizer_, text):
    sentences = []
    for sentence in tokenizer_.sentence_tokenize(text):
        sentences.append(sentence)
    return sentences


def get_words(tokenizer_, review, **kwargs):
    return tokenizer_.tokenize(review, **kwargs)


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--input', type=GzipFileType('r'), default=[sys.stdin], nargs='*',
                        help='Input file (in TSV format, optionally compressed)')
    parser.add_argument('--lemmatize', action='store_true',
                        help='lemmatize words (overrides --stem)')
    parser.add_argument('--stem', action='store_true',
                        help='stem words')
    parser.add_argument('--sentences', action='store_true',
                        help='split by sentence instead of by record')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process this many lines (for testing)')
    parser.add_argument('--output', type=GzipFileType('w'), default=sys.stdout,
                        help='File to write sentences to, optionally compressed')
    namespace = parser.parse_args(args)
    return namespace


def get_review_iterator(args):
    return get_reviews(*args.input)


def get_tokenizer(args):
    if args.lemmatize:
        tokenizer = SimpleSentenceTokenizer(lemmatizer=wordnet.WordNetLemmatizer())
    elif args.stem:
        tokenizer = SimpleSentenceTokenizer(stemmer=PorterStemmer())
    else:
        tokenizer = SimpleSentenceTokenizer()
    return tokenizer


def run(args):
    tokenizer = get_tokenizer(args)
    iterator = get_review_iterator(args)
    if args.limit:
        iterator = islice(iterator, args.limit)
    if args.sentences:
        tokenize = get_sentences
    else:
        tokenize = get_words
    for record in Parallel(n_jobs=1)(delayed(tokenize)(tokenizer, review) for review in iterator):
        write_json_line(args.output, record)


if __name__ == "__main__":
    run(parse_args())
