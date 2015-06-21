import unittest
from tests import count_prefix
from functools import partial
from flaubert.preprocess import TOKENIZER
from flaubert.tokenize import RegexpFeatureTokenizer
from pymaptools.utils import SetComparisonMixin


class TestFeatureTokens(unittest.TestCase, SetComparisonMixin):

    maxDiff = 2000

    def setUp(self):
        self.tokenize = partial(TOKENIZER.tokenize, remove_stopwords=False)
        self.base_tokenizer = RegexpFeatureTokenizer(debug=True)

    def test_western_emoticons_happy(self):
        """With custom features removed, this text should be idempotent on tokenization
        """
        text = u":-) :) =) =)) :=) >:) :] :') :^) (: [: ((= (= (=: <3 :-p :D :o"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not token.startswith(u"<EMOTIC"))
        self.assertEqual(text.lower(), reconstructed)
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(36, count_prefix(u"EMOTIC", group_names))

    def test_western_emoticons_sad(self):
        """With custom features removed, this text should be idempotent on tokenization
        """
        text = u":-( :( =( =(( :=( >:( :[ :'( :^( ): ]: ))= )= )=: :-c :C :O"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not token.startswith(u"<EMOTIC"))
        self.assertEqual(text.lower(), reconstructed)

        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(34, count_prefix(u"EMOTIC", group_names))

    def test_no_emoticon(self):
        """No emoticon should be detected in this text
        """
        text = u"(8) such is the game): -  (7 or 8) and also (8 inches)" \
            u" and spaces next to parentheses ( space ) ."
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(0, count_prefix(u"EMOTIC", group_names))

    def test_eastern_emoticons(self):
        text = u"*.* (^_^) *_* *-* +_+ ~_~"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not (token.startswith(u"<") and token.endswith(u">")))
        self.assertEqual(text, reconstructed)
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(6, count_prefix(u"EMOTIC", group_names))

    def test_russian_emoticons(self):
        text = u"haha! ))))) )) how sad (("
        tokens = self.tokenize(text)
        reconstructed = u' '.join(tokens)
        self.assertEqual(u'haha ! ))) )) how sad ((', reconstructed)
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(3, count_prefix(u"EMOTIC", group_names))

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
            [u'a', u'dummy', u'comment', u'with', u'<URI>', u'and', u'<EMAIL>'],
            tokens)

    def test_rating_1(self):
        text = u"which deserves 11 out of 10,"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<11 / 10>'], tokens)

    def test_rating_2(self):
        text = u"I give this film 10 stars out of 10."
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<10 / 10>'], tokens)

    def test_rating_3(self):
        text = u"A must-see for fans of Japanese horror.10 out of 10."
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<10 / 10>'], tokens)

    def test_rating_4(self):
        text = u"a decent script.<br /><br />3/10"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<3 / 10>'], tokens)

    def test_rating_5(self):
        text = u"give it five stars out of ten"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<5 / 10>'], tokens)

    def test_rating_6(self):
        text = u"give it 3 1/2 stars out of five"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<7 / 10>'], tokens)

    def test_rating_7(self):
        text = u"give it ** 1/2 stars out of four"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<6 / 10>'], tokens)

    def test_rating_8(self):
        text = u"has been done so many times.. 7 of 10"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<7 / 10>'], tokens)

    def test_rating_9(self):
        text = u"has been done so many times.. 8 / 10"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<8 / 10>'], tokens)

    def test_rating_10(self):
        text = u"I give it a 7 star rating"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<7 / 10>'], tokens)

    def test_rating_11(self):
        text = u"Grade: * out of *****"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<2 / 10>'], tokens)

    def test_grade_1(self):
        text = u"can save this boring, Grade B+ western."
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<GRADE_B+>'], tokens)

    def test_grade_2(self):
        text = u"can save this boring, Grade B western."
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<GRADE_B>'], tokens)

    def test_grade_3(self):
        text = u"My grade: F."
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<GRADE_F>'], tokens)

    def test_grade_4(self):
        text = u"mindless B-grade \"entertainment.\""
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<GRADE_B>'], tokens)

    def test_decade_1(self):
        text = u"Nice 1950s & 60s \"Americana\""
        tokens = self.tokenize(text)
        self.assertSetContainsSubset(
            [u'nice', u'1950', u'60', u'americana'], tokens)

    def test_decade_2(self):
        text = u"Nice 1950s & 60's \"Americana\""
        tokens = self.tokenize(text)
        self.assertSetContainsSubset(
            [u'nice', u'1950', u'60', u'americana'], tokens)
