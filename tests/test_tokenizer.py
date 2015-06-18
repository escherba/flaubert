import unittest
from tests import count_prefix
from functools import partial
from nl2vec.preprocess import TOKENIZER
from nl2vec.tokenize import RegexFeatureTokenizer
from pymaptools.utils import SetComparisonMixin


class TestFeatureTokens(unittest.TestCase, SetComparisonMixin):

    maxDiff = 2000

    def setUp(self):
        self.tokenize = partial(TOKENIZER.tokenize, remove_stopwords=False)
        self.base_tokenizer = RegexFeatureTokenizer(debug=True)

    def test_western_emoticons(self):
        """With custom features removed, this text should be idempotent on tokenization
        """
        text = u":-) :) =) =)) :=) >:) :] :') :^) (: [: ((= (= (=: <3 :-p :d :o"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not token.startswith(u"<EMOTICON"))
        self.assertEqual(text, reconstructed)

        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(18, count_prefix(u"EMOTICON", group_names))

    def test_no_emoticon(self):
        """No emoticon should be detected in this text
        """
        text = u"(8) such is the game): -  (7 or 8) and also (8 inches)" \
            u" and spaces next to parentheses ( space ) ."
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(0, count_prefix(u"EMOTICON", group_names))

    def test_eastern_emoticons(self):
        text = u"*.* (^_^) *_* *-* +_+ ~_~"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not (token.startswith(u"<") and token.endswith(u">")))
        self.assertEqual(text, reconstructed)
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(6, count_prefix(u"EMOTICON", group_names))

    def test_russian_emoticons(self):
        text = u"haha! ))))) )) how sad (("
        tokens = self.tokenize(text)
        reconstructed = u' '.join(tokens)
        self.assertEqual(u'haha ! ))) )) how sad ((', reconstructed)
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(3, count_prefix(u"EMOTICON", group_names))

    def test_ascii_arrow(self):
        text = u"Look here -->> such doge <<<"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset(
            {'<ASCIIARROW_RIGHT>', '<ASCIIARROW_LEFT>'}, tokens)

    def test_abbrev(self):
        text = u"S&P index of X-men in the U.S."
        tokens = self.tokenize(text)
        self.assertListEqual(
            [u's&p', u'index', u'of', u'x-men', u'in', u'the', u'u.s.'],
            tokens)

    def test_url_email(self):
        text = u"a dummy comment with http://www.google.com/ and sergey@google.com"
        tokens = self.tokenize(text)
        self.assertListEqual(
            [u'a', u'dummy', u'comment', u'with', u'<uri>', u'and', u'<email>'],
            tokens)
