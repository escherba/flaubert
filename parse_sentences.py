import nltk
import pandas as pd
import unicodedata
import regex as re
from pymaptools.io import write_json_line, PathArgumentParser, GzipFileType
from bs4 import BeautifulSoup
from functools import partial
from nltk.corpus import stopwords
from nltk.stem import wordnet, PorterStemmer
from nltk import pos_tag
from joblib import Parallel, delayed

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

    _sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def __init__(self, word_regex=u"\\p{L}+", form='NFKC', stop_words=stopwords.words("english"),
                 lemmatizer=None, stemmer=None):
        self._normalize = partial(unicodedata.normalize, form)
        self._word_tokenize = re.compile(word_regex, re.UNICODE | re.IGNORECASE).findall
        self._stopwords = frozenset(stop_words)
        self._sentence_tokenize = self._sentence_tokenizer.tokenize
        self._lemmatizer = lemmatizer
        self._stemmer = stemmer
        self._pos_tag = pos_tag

    def tokenize(self, text, remove_html=True, remove_stopwords=False):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
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
            words = self.tokenize(raw_sentence, remove_html=False)
            if not words:
                continue
            sentences.append(words)
        return sentences


def read_review_data(filename):
    return pd.read_csv(filename, header=0, delimiter="\t", quoting=2, escapechar="\\", quotechar='"', encoding="utf-8")



tokenizer1 = SimpleSentenceTokenizer(lemmatizer=wordnet.WordNetLemmatizer())
tokenizer2 = SimpleSentenceTokenizer(stemmer=PorterStemmer())
tokenizer3 = SimpleSentenceTokenizer()


def review_to_wordlist(review, **kwargs):
    return tokenizer3.tokenize(review, **kwargs)


train = read_review_data("labeledTrainData.tsv")
test = read_review_data("testData.tsv")
unlabeled_train = read_review_data("unlabeledTrainData.tsv")


def get_reviews(*datasets):
    for dataset in datasets:
        for review in dataset['review']:
            yield review


def get_sentences(tokenizer, text):
    sentences = []
    for sentence in tokenizer.sentence_tokenize(text):
        sentences.append(sentence)
    return sentences


sent_tokenize = partial(get_sentences, tokenizer1)
review_iterator = get_reviews(train, unlabeled_train)
sentences = Parallel(n_jobs=6)(delayed(sent_tokenize)(sent) for sent in review_iterator)


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--input', type=str, metavar='FILE', required=True,
                        help='Input CSV')
    parser.add_argument('--output', type=GzipFileType('w'), required=True,
                        help='File to write sentences to')
    namespace = parser.parse_args(args)
    return namespace


def run(args):
    with open(args.output, 'w') as fh:
        for sentence in sentences:
            write_json_line(fh, sentence)


if __name__ == "__main__":
    run(parse_args())
