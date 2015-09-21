import unittest
from tests import count_prefix
from functools import partial
from collections import Counter
from flaubert.preprocess import TOKENIZER
from flaubert.tokenize import RegexpFeatureTokenizer
from pymaptools.utils import SetComparisonMixin


class TestFeatureTokens(unittest.TestCase, SetComparisonMixin):

    maxDiff = 2000

    def setUp(self):
        self.tokenizer = TOKENIZER
        self.tokenize = partial(TOKENIZER.tokenize, remove_stopwords=False)
        self.sentence_tokenize = TOKENIZER.sentence_tokenize
        self.base_tokenizer = RegexpFeatureTokenizer(debug=True)

    def test_preprocess(self):
        text = u"wow \u2014 such \u2013 doge"
        preprocessed = self.tokenizer.preprocess(text)
        self.assertEqual(u'wow --- such -- doge', preprocessed)

    def test_dashes(self):
        text = u"wow \u2014 such \u2013 doge -- and --- are dashes"
        counts = Counter(self.tokenize(text))
        self.assertEqual(2, counts[u'--'])
        self.assertEqual(2, counts[u'---'])

    def test_censored(self):
        text = u"she's a b*tch in a f***d world"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'she', u"'s", u'b*tch', u'f***d'], tokens)

    def test_sentence_split_ellipsis(self):
        """
        Make sure there is a sentence break after ellipsis

        Note: The sentence splitter we use does not treat ellipsis as a
        sentence terminator if the word after it is not capitalized.
        """
        text = u"I had a feeling that after \"Submerged\", this one wouldn't " \
               u"be any better... I was right."
        sentences = self.sentence_tokenize(text)
        self.assertEqual(2, len(sentences))

    def test_sentence_split_br(self):
        """
        Make sure there is a sentence break before "O.K."
        """
        text = u'Memorable lines like: "You son-of-a-gun!", "You son-of-a-witch!",' \
               u' "Shoot!", and "Well, Forget You!"<br /><br />O.K. Bye.'
        sentences = self.sentence_tokenize(text)
        joint = u' | '.join([u' '.join(sentence) for sentence in sentences])
        self.assertIn(u' | o', joint)

    def test_western_emoticons_happy(self):
        """With custom features removed, this text should be idempotent on tokenization
        """
        text = u":-) :) =) =)) :=) >:) :] :') :^) (: [: ((= (= (=: :-p :D :o"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not token.startswith(u"<EMOTIC"))
        self.assertEqual(text.lower(), reconstructed)
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(34, count_prefix(u"EMOTIC", group_names))

    def test_western_emoticons_sad(self):
        """With custom features removed, this text should be idempotent on tokenization
        """
        text = u":-( :( =( =(( :=( >:( :[ :'( :^( ): ]: ))= )= )=: :-c :C :O :@"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not token.startswith(u"<EMOTIC"))
        self.assertEqual(text.lower(), reconstructed)

        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(35, count_prefix(u"EMOTIC", group_names))

    def test_hearts(self):
        """With custom features removed, this text should be idempotent on tokenization
        """
        text = u"<3 full heart </3 heartbreak"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not token.startswith(u"<EMOTIC"))
        self.assertEqual(text.lower(), reconstructed)

        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertSetContainsSubset([u'<3', u'<EMOTIC_HEART_HAPPY>', u'</3', u'<EMOTIC_HEART_SAD>'],
                                     tokens)
        self.assertEqual(4, count_prefix(u"EMOTIC", group_names))

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

    def test_contraction(self):
        text = u"Daniel's life isn't great"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'daniel', u"'s", u'be', u"n't"], tokens)

    def test_contraction_lookalike(self):
        text = u"abr'acad'a'bra"
        tokens = self.tokenize(text)
        self.assertEqual(text, u"'".join(tokens))

    def test_special_3d(self):
        text = u"3-d (3D) effect"
        tokens = self.tokenize(text)
        self.assertListEqual([u"<3D>", u"<3D>", u"effect"], tokens)

    def test_rating_false_0(self):
        text = u"I re-lived 1939/40 and my own evacuation from London"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'40'], tokens)

    def test_rating_false_1(self):
        text = u"Update: 9/4/07-I've now read Breaking Free"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<DATE>'], tokens)

    def test_rating_false_2(self):
        text = u"the humility of a 10 year old in cooking class"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'10'], tokens)

    def test_rating_0(self):
        text = u"My rating: 8.75/10----While most of this show is good"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<9/10>'], tokens)

    def test_rating_1(self):
        text = u"which deserves 11 out of 10,"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<11/10>'], tokens)

    def test_rating_2(self):
        text = u"I give this film 10 stars out of 10."
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<10/10>'], tokens)

    def test_rating_3(self):
        text = u"A must-see for fans of Japanese horror.10 out of 10."
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<10/10>'], tokens)

    def test_rating_4(self):
        text = u"a decent script.<br /><br />3/10"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<3/10>'], tokens)

    def test_rating_5(self):
        text = u"give it five stars out of ten"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<5/10>'], tokens)

    def test_rating_6(self):
        text = u"give it 3 1/2 stars out of five"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<7/10>'], tokens)

    def test_rating_7(self):
        text = u"give it ** 1/2 stars out of four"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<6/10>'], tokens)

    def test_rating_8(self):
        text = u"has been done so many times.. 7 of 10"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<7/10>'], tokens)

    def test_rating_9(self):
        text = u"has been done so many times.. 8 / 10"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<8/10>'], tokens)

    def test_rating_10(self):
        text = u"I give it a 7 star rating"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<7/10>'], tokens)

    def test_rating_11(self):
        text = u"Grade: * out of *****"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<2/10>'], tokens)

    def test_rating_12(self):
        text = u"Final Judgement: **/****"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<5/10>'], tokens)

    def test_rating_13(self):
        text = u'on March 18th, 2007.<br /><br />84/100 (***)'
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<8/10>'], tokens)

    def test_rating_14(self):
        text = u'I give it a full 10.'
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<10/10>'], tokens)

    def test_rating_15(self):
        text = u'I give it a -50 out of 10. MY GOD!!!!'
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<0/10>'], tokens)

    def test_rating_16(self):
        text = u"* * 1/2 / * * * *"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<6/10>'], tokens)

    def test_rating_17(self):
        text = u"i gave this movie a 2 for the actors"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<2/10>'], tokens)

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
            [u'nice', u'1950s', u'60s', u'americana'], tokens)

    def test_decade_2(self):
        text = u"Nice 1950s & 60's \"Americana\""
        tokens = self.tokenize(text)
        self.assertSetContainsSubset(
            [u'nice', u'1950s', u'60s', u'americana'], tokens)

    def test_mention(self):
        text = u"@RayFranco is answering to @AnPel, this is a real '@username83' " \
               u"but this is an@email.com, and this is a @probablyfaketwitterusername"
        token_counts = Counter(self.tokenize(text))
        self.assertEqual(4, token_counts['<MENTION>'])
        self.assertEqual(1, token_counts['<EMAIL>'])

    def test_emphasis(self):
        text = "@hypnotic I know  *cries*"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<EMPHASIS>', u'cry'], tokens)
