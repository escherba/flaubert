# -*- coding: utf-8 -*-

import regex as re
from functools import partial
from collections import deque
from pymaptools.inspect import get_object_attrs

FIND_STARS = re.compile(u'\\*').findall
STRIP_NONWORD = partial(re.compile(u'\\W+').sub, u'')
STRIP_SPACES = partial(re.compile(u'\\s+').sub, u'')

NUM2DEC = {
    u'zero': 0,
    u'one': 1,
    u'two': 2,
    u'three': 3,
    u'four': 4,
    u'five': 5,
    u'six': 6,
    u'seven': 7,
    u'eight': 8,
    u'nine': 9,
    u'ten': 10,
}

# contractions like can't, I'd, he'll, I'm, I've
EN_APO_CONTRACTIONS = frozenset([
    u'll', u'd', u're', u't', u'm', u've', u's'
])

CONTRACTION_FIRST_MAP = {
    u"wo": u'will',
    u"sha": u'shall',
    u"ca": u"can",
    u"ai": u"am"
}


def count_stars(num_stars):
    dec_num_stars = NUM2DEC.get(num_stars)
    if dec_num_stars is None:
        dec_num_stars = len(FIND_STARS(num_stars))
        if dec_num_stars == 0:
            dec_num_stars = int(num_stars)
    return float(dec_num_stars)


DEFAULT_FEATURE_MAP = u"""
(?P<SPECIAL>\\b__([A-Za-z]+)__\\b)
|
(?P<CENSORED>\\b\\p{L}+(?:\\*+\\p{L}+)+\\b)
|
(?P<TIMEOFDAY>\\b[0-2]?[0-9]:[0-6][0-9](?:\\s*[AaPp][Mm])?\\b)
|
(?P<DATE>\\b[0-9]+\\s*\\/\\s*[0-9]+\\s*\\/\\s*[0-9]+\\b)
|
(?P<EMOTIC_EAST_LO>\\(?[\\+\\^ˇ\\*\\->~][_\\.][\\+\\^ˇ\\*\\-<~]\\)?)
|
(?P<EMOTIC_EAST_HI>\\(?[\\^ˇ\\*][\\-~oO][\\^ˇ\\*]\\)?)
|
(?P<EMOTIC_WEST_LEFT>\\>?(?:=|(?:[:;]|(?<![\\w\\(\\)])[Bb])[\\-=\\^']?)([\\(\\)\\*\\[\\]\\|]+|[cCoOpPdD]\\b))
|
(?P<EMOTIC_WEST_RIGHT>(?<!\\w)([\\(\\)\\[\\]\\|]+)(?:(?:[\\-=\\^'][:;]?|[\\-=\\^']?[:;])(?![\\w\\(\\)])|=))
|
(?P<EMOTIC_WEST_LEFT_HAPPY>(?<![0-9])[:;]3+\\b)
|
(?P<EMOTIC_RUSS_HAPPY>\\){2,})
|
(?P<EMOTIC_RUSS_SAD>\\({2,})
|
(?P<EMOTIC_HEART>(?<![0-9])\\<(\\/?)3+\\b)
|
(?P<CONTRACTION>\\b([a-zA-Z]+)'([a-zA-Z]{1,2})\\b)     # you're, it's, it'd, he'll
|
(?P<STARRATING>((?:(?:\\s\\-)?[0-9]{1,2})|(?:\\*\\s?)+|%(number)s)(\\.[0-9]+|[\\s-]*[1-9]\\s?\\/\\s?[1-9])?\\s*(?:stars?(?:\\s+rating)?)?\\s*(?:\\/\\s*|\\(?(?:out\\s+)?of\\s+)((?:4|5|100?|four|five|ten)\\b|(?:\\*\\s?)+)(?:\\s+stars)?)
|
(?P<STARRATING_FULL>\\bfull\\s10\\b)
|
(?P<STARRATING_X>\\b(?:a|my)\\s+(\\-?[0-9](?:\\.[0-9])?)[\\s-]+(?:star\\s+)?(?:rating|for)\\b)
|
(?P<MPAARATING>pg[-\\s]?13|nc[-\\s]?17)
|
(?P<GRADE_POST>\\bgrade\\s*[:-]?\\s*([a-f](?:\\+|\\b)))
|
(?P<GRADE_PRE>\\b([a-f]\\+?)\\s*-?\\s*grade\\b)
|
(?P<THREED>\\b3\\-?d\\b)
|
(?P<CURRENCY>(?<=\\s)\\p{Sc}+(?=(?:\\s|[0-9])))
|
(?P<DECADE>\\b((?:18|19|20)?[0-9]0)(?:\\s*'\\s*)?s\\b)
|
(?P<ASCIIARROW_RIGHT>([\\-=]?\\>{2,}|[\\-=]+\\>))        # -->, ==>, >>, >>>
|
(?P<ASCIIARROW_LEFT>(\\<{2,}[\\-=]?|\\<[\\-=]+))         # <<<, <<, <==, <--
|
(?P<MNDASH>\\s\\-\\s|\\-{2,3}|\\u2013|\\u2014)
|
(?P<ABBREV1>\\b\\p{L}([&\\/\\-\\+])\\p{L}\\b)            # entities like S&P, w/o
|
(?P<ABBREV2>\\b(\\p{L})-(\\p{L}{2,})\\b)                 # X-Factor, X-men, T-trak
|
(?P<ABBREV3>\\b(?:\\p{L}\\.){2,})                        # abbreviation with periods like U.S.
|
(?P<ELLIPSIS>(?:\\.\\s*){2,}|\\u2026)                    # ellipsis
|
(?::+(?!\\/\\/)|[!?¡¿]+|[,\\.])                          # punctuation
|
(\\w+)                                                # any non-zero sequence of letters
""" % dict(
    number=u'|'.join(NUM2DEC.keys())
)


class RegexpFeatureTokenizer(object):
    """A regex-based feature extractor and tokenizer

    This feature extractor is meant to be run after basic preprocessing
    has been applied (strip out HTML, normalize Unicode etc). It works
    by matching a series of regex groups and then dispatching appropriately
    named methods based on matches.

    If the last matched group is unnamed, we return the matched content as is.
    If the last group has a name, we look it up among method names that start
    with handle_ and one exists, pass it to it. Otherwise we annotate the
    match with the matched name (hierarchically if the name is multi-part)
    and also return the content of the match.

    Hashtag and user mention regexes were writen by Space who based them on
    https://github.com/twitter/twitter-text/blob/master/js/twitter-text.js
    For hashtags, mentions, and cashtags, we generate a plain word token as
    well as one with the appropriate character prefixed.
    Note that hashtag are supposed to be prefixed by space or by string boundary,
    however this doesn't capture user intent (users may concatenate hashtags
    together).

    More detailed description of some features:

    * A hashtag is made of a sequence of alpha-numeric characters  with at least
    one alphabet character, e.g. "we're #1" has no tag. Note, there are two
    different Unicode characters that can serve as prefixes: x23 \\uff03

    * A twitter mention is all in ASCII alpha-numeric. Note there are two different
    Unicode characters that can serve as prefixes: x40 \\uff20

    * A stock symbol is a 1-4 letter word with optional fifth (actually last letter),
    delimited by a period, that signifies security type (e.g. $BRK.A, $BRK.B)
    http://www.investopedia.com/ask/answers/06/nasdaqfifthletter.asp
    """

    dispatch_prefix = 'handle_'

    def __init__(self,
                 pattern=DEFAULT_FEATURE_MAP,
                 groupname_format=u"<%s>",
                 word_buffer_len=5,
                 flags=re.UNICODE | re.DOTALL | re.VERBOSE,
                 debug=False):

        # attributes equal to parameter names
        self.regex = re.compile(pattern, flags)
        self.groupname_format = groupname_format
        self.word_buffer_len = word_buffer_len

        self.wrap_result = (lambda x, y: (x, y)) if debug else (lambda x, y: x)
        self.dispatch_map = self.create_feature_dispatcher()

    def create_feature_dispatcher(self):
        result = {}
        prefix = self.dispatch_prefix
        prefix_offset = len(prefix)
        for method in get_object_attrs(self):
            if method.startswith(prefix):
                result[method[prefix_offset:].upper()] = getattr(self, method)
        return result

    def _group_name(self, match):
        return self.groupname_format % match.lastgroup

    def _group_tag(self, match, *args):
        yield self._group_name(match)

    handle_asciiarrow_left = _group_tag
    handle_asciiarrow_right = _group_tag
    handle_timeofday = _group_tag
    handle_date = _group_tag

    def handle_special(self, match, *args):
        tag_name = match.group(match.lastindex + 1).upper()
        yield self.groupname_format % tag_name

    def handle_contraction(self, match, *args):
        """Words with a single apostrophe in them
        """
        first = match.group(match.lastindex + 1)
        second = match.group(match.lastindex + 2)
        if second in EN_APO_CONTRACTIONS:
            if second == u"t" and first[-1] == u"n":
                firstp = first[0:-1]
                yield CONTRACTION_FIRST_MAP.get(firstp, firstp)
                yield u"n't"
            else:
                yield first
                yield u"'" + second
        else:
            yield first
            yield second

    def handle_emotic_west_left(self, match, *args):
        mouth = match.group(match.lastindex + 1)[0]
        group_name = match.lastgroup
        if mouth in u")]pdPD":
            group_name = u'_'.join([group_name, 'HAPPY'])
        elif mouth in u"([cC":
            group_name = u'_'.join([group_name, 'SAD'])
        yield match.group()
        yield self.groupname_format % group_name

    def handle_emotic_west_right(self, match, *args):
        mouth = match.group(match.lastindex + 1)[-1]
        group_name = match.lastgroup
        if mouth in u"([":
            group_name = u'_'.join([group_name, 'HAPPY'])
        elif mouth in u")]":
            group_name = u'_'.join([group_name, 'SAD'])
        yield match.group()
        yield self.groupname_format % group_name

    def handle_emotic_heart(self, match, *args):
        group_name = match.lastgroup
        broken = match.group(match.lastindex + 1)
        if broken:
            group_name = u'_'.join([group_name, 'SAD'])
        else:
            group_name = u'_'.join([group_name, 'HAPPY'])
        yield match.group()
        yield self.groupname_format % group_name

    def handle_starrating(self, match, *args):
        """
        Convert miscellaneous ways to write a star rating into something like
        4/10
        """
        num_stars = count_stars(match.group(match.lastindex + 1))
        modifier = match.group(match.lastindex + 2)
        out_of = count_stars(match.group(match.lastindex + 3))
        if modifier:
            if u'/' in modifier:
                numer, denom = modifier.split(u'/')
                numer, denom = float(numer), float(denom)
                modifier = numer / denom
            else:
                modifier = float(modifier)
            if num_stars >= 0.0:
                num_stars += modifier
            else:
                num_stars -= modifier
        num_stars *= (10.0 / out_of)
        num_stars = max(0, min(11, int(round(num_stars))))
        yield u"<%d/%d>" % (num_stars, 10)

    def handle_starrating_x(self, match, *args):
        num_stars = int(round(float(match.group(match.lastindex + 1))))
        num_stars = max(0, min(11, num_stars))
        yield u"<%d/%d>" % (num_stars, 10)

    def handle_starrating_full(self, match, *args):
        yield u"<10/10>"

    def wrapped_entity_handler(self, match, *args):
        yield self.groupname_format % STRIP_NONWORD(match.group()).upper()

    handle_mpaarating = wrapped_entity_handler
    handle_threed = wrapped_entity_handler

    def handle_decade(self, match, *args):
        yield STRIP_NONWORD(match.group()).lower()

    def handle_currency(self, match, *args):
        yield match.group()[0]

    def grade_handler(self, match, *args):
        grade = match.group(match.lastindex + 1).upper()
        yield self.groupname_format % u'_'.join([u"GRADE", grade])

    handle_grade_pre = grade_handler
    handle_grade_post = grade_handler

    def handle_mndash(self, match, *args):
        extracted = match.group()
        if extracted == u'\u2013':
            yield u'--'
        elif extracted == u'\u2014':
            yield u'---'
        else:
            yield STRIP_SPACES(extracted)

    def handle_ellipsis(self, match, *args):
        extracted = match.group()
        if extracted == u"\u2026":
            yield u"..."
        else:
            yield STRIP_NONWORD(extracted)

    def __call__(self, text):
        return self.tokenize(text)

    def tokenize(self, text):
        return list(self.tokenize_iter(text))

    def tokenize_iter(self, text):
        dispatch_map = self.dispatch_map
        seen_words = deque(maxlen=self.word_buffer_len)
        wrap_result = self.wrap_result
        for match in self.regex.finditer(text):
            group_name = match.lastgroup
            method = dispatch_map.get(group_name)
            if method is None:
                word = match.group()
                seen_words.append(word)
                yield wrap_result(word, match)
            else:
                for word in method(match, text, seen_words):
                    seen_words.append(word)
                    yield wrap_result(word, match)
