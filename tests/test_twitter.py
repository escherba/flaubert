import unittest
from tests import count_prefix
from functools import partial
from collections import Counter
from flaubert.preprocess import tokenizer_builder
from flaubert.tokenize import RegexpFeatureTokenizer
from pymaptools.utils import SetComparisonMixin

FEATURES = [
    'CUSTOMTOKEN', 'REPLY', 'CENSORED', 'EMPHASIS_B', 'EMPHASIS_U', 'TIMEOFDAY',
    'DATE', 'EMOTIC_EAST_LO', 'EMOTIC_EAST_HI', 'EMOTIC_EAST_SAD', 'EMOTIC_WEST_L',
    'EMOTIC_WEST_R', 'EMOTIC_WEST_L_HAPPY', 'EMOTIC_WEST_CHEER', 'EMOTIC_WEST_L_MISC',
    'EMOTIC_WEST_R_MISC', 'EMOTIC_RUSS_HAPPY', 'EMOTIC_RUSS_SAD', 'EMOTIC_HEART',
    'CONTRACTION', 'MPAARATING', 'GRADE_POST', 'GRADE_PRE', 'THREED', 'NUMBER',
    'HASHTAG', 'MENTION', 'ASCIIARROW_R', 'ASCIIARROW_L', 'MNDASH', 'ABBREV1',
    'ABBREV2', 'ABBREV3', 'ELLIPSIS', 'XOXO', 'XX', 'PUNKT', 'ANYWORD']


class TestTwitterTokens(unittest.TestCase, SetComparisonMixin):

    maxDiff = 2000

    def setUp(self):
        TOKENIZER = tokenizer_builder(features=FEATURES)
        self.tokenizer = TOKENIZER
        self.tokenize = partial(TOKENIZER.tokenize, remove_stopwords=False)
        self.sentence_tokenize = TOKENIZER.sentence_tokenize
        self.base_tokenizer = RegexpFeatureTokenizer(features=FEATURES, debug=True)

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
        self.assertEqual(len(tokens), count_prefix(u"EMOTIC", group_names))

    def test_western_emoticons_sad(self):
        """With custom features removed, this text should be idempotent on tokenization
        """
        text = u":-( :( =( =(( :=( >:( :[ :'( :^( ): ]: ))= )= )=: :-c :C :O :@ D:"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not token.startswith(u"<EMOTIC"))
        self.assertEqual(text.lower(), reconstructed)

        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(len(tokens), count_prefix(u"EMOTIC", group_names))

    def test_western_emoticons_misc(self):
        """With custom features removed, this text should be idempotent on tokenization
        """
        text = u":0 :l :s :x \o/ \m/"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u':0', u':l', u':s', u':x', u'\o/', u'\m/'], tokens)

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
        self.assertEqual(len(tokens) - 3, count_prefix(u"EMOTIC", group_names))

    def test_no_emoticon(self):
        """No emoticon should be detected in this text
        """
        text = u"(8) such is the game): -  (7 or 8) and also (8 inches)" \
            u" and spaces next to parentheses ( space ) ."
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(0, count_prefix(u"EMOTIC", group_names))

    def test_eastern_emoticons(self):
        text = u"*.* (^_^) *_* *-* +_+ ~_~ -.- -__- -___- t_t q_q ;_; t.t q.q ;.;"
        tokens = self.tokenize(text)
        reconstructed = u' '.join(token for token in tokens if not (token.startswith(u"<") and token.endswith(u">")))
        self.assertEqual(text, reconstructed)
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(len(tokens), count_prefix(u"EMOTIC", group_names))

    def test_russian_emoticons(self):
        text = u"haha! ))))) )) how sad (("
        tokens = self.tokenize(text)
        reconstructed = u' '.join(tokens)
        self.assertEqual(u'haha ! ))) )) how sad ((', reconstructed)
        group_names = [m.lastgroup for m in zip(*self.base_tokenizer.tokenize(text))[1]]
        self.assertEqual(len(tokens) - 4, count_prefix(u"EMOTIC", group_names))

    def test_ascii_arrow(self):
        text = u"Look here -->> such doge <<<"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset(
            {'<ASCIIARROW_R>', '<ASCIIARROW_L>'}, tokens)

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

    def test_emphasis_star(self):
        text = u"@hypnotic I know  *cries*"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<EMPHASIS_B>', u'cry'], tokens)

    def test_emphasis_underscore(self):
        text = u"I _hate_ sunblock"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<EMPHASIS_U>', u'hate'], tokens)

    def test_unescape(self):
        text = u"@artmeanslove I &lt;3 that book"
        tokens = self.tokenize(text)
        self.assertSetContainsSubset([u'<3', u'<EMOTIC_HEART_HAPPY>'], tokens)

    def test_kisses(self):
        text = u"ohh lovely  x hehe x count naked people hehe that's " \
            u"what you always tell me to do hehe x x x night night xxxxx"
        token_counts = Counter(self.tokenize(text))
        self.assertEqual(2, token_counts[u'<XX>'])

    def test_kisses_hugs(self):
        text = u"all right, xo xo vou mimi now xoxo"
        token_counts = Counter(self.tokenize(text))
        self.assertEqual(2, token_counts[u'<XOXO>'])
