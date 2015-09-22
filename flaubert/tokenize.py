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


DEFAULT_FEATURE_MAP = [
    ('CUSTOMTOKEN', u"""\\b__([A-Za-z]+)__\\b"""),

    ('REPLY', u"""^\\s*\\.?\\s*(?=@)"""),

    ('CENSORED', u"""\\b\\p{L}+(?:\\*+\\p{L}+)+\\b"""),

    ('EMPHASIS_B', u"""\\*+(\\p{L}+)\\*+"""),

    ('EMPHASIS_U', u"""(?<![$#@])\\b_+(\\p{L}+)_+\\b"""),

    ('TIMEOFDAY', u"""\\b[0-2]?[0-9]:[0-6][0-9](?:\\s*[AaPp][Mm])?\\b"""),

    ('DATE', u"""\\b[0-9]+\\s*\\/\\s*[0-9]+\\s*\\/\\s*[0-9]+\\b"""),

    ('EMOTIC_EAST_LO', u"""\\(?[\\+\\^ˇ\\*\\->~](?:_+|\\.)[\\+\\^ˇ\\*\\-<~]\\)?"""),

    ('EMOTIC_EAST_HI', u"""\\(?[\\^ˇ\\*][\\-~oO][\\^ˇ\\*]\\)?"""),

    ('EMOTIC_EAST_SAD', u"""\\b[tTqQ][_\\.][tTqQ]\\b|;[_\\.];"""),

    ('EMOTIC_WEST_L', u"""\\>?(?:=|(?:[:;]|(?<![\\w\\(\\)])[Bb])[\\-=\\^']?)([\\(\\)\\*\\[\\]\\|]+|[cCoOpPdDsSlL0xX]\\b)"""),

    ('EMOTIC_WEST_R', u"""(?<!\\w)([dD]|[\\(\\)\\[\\]\\|]+)(?:(?:[\\-=\\^'][:;]?|[\\-=\\^']?[:;])(?![\\w\\(\\)])|=)"""),

    ('EMOTIC_WEST_L_HAPPY', u"""(?<![0-9])[:;]3+\\b"""),

    ('EMOTIC_WEST_CHEER', u"""\\\[om]/"""),

    ('EMOTIC_WEST_L_MISC', u"""(?<![^\\p{P}\\s])[:=]([$@\\\/])(?![^\\s\\p{P}])"""),

    ('EMOTIC_WEST_R_MISC', u"""(?<![^\\p{P}\\s])([$@\\\/])[:=](?![^\\s\\p{P}])"""),

    ('EMOTIC_RUSS_HAPPY', u"""\\){2,}"""),

    ('EMOTIC_RUSS_SAD', u"""\\({2,}"""),

    ('EMOTIC_HEART', u"""(?<![0-9])\\<(\\/?)3+\\b"""),

    ('CONTRACTION', u"""\\b([a-zA-Z]+)'([a-zA-Z]{1,2})\\b"""),  # you're, it's, it'd, he'll,

    ('STARRATING', u"""((?:(?:\\s\\-)?[0-9]{1,2})|(?:\\*\\s?)+|%(number)s)(\\.[0-9]+|[\\s-]*[1-9]\\s?\\/\\s?[1-9])?\\s*(?:stars?(?:\\s+rating)?)?\\s*(?:\\/\\s*|\\(?(?:out\\s+)?of\\s+)((?:4|5|100?|four|five|ten)\\b|(?:\\*\\s?)+)(?:\\s+stars)?""" % dict(number=u'|'.join(NUM2DEC.keys()))),

    ('STARRATING_FULL', u"""\\bfull\\s10\\b"""),

    ('STARRATING_X', u"""\\b(?:a|my)\\s+(\\-?[0-9](?:\\.[0-9])?)[\\s-]+(?:star\\s+)?(?:rating|for)\\b"""),

    ('MPAARATING', u"""pg[-\\s]?13|nc[-\\s]?17"""),

    ('GRADE_POST', u"""\\bgrade\\s*[:-]?\\s*([a-f](?:\\+|\\b))"""),

    ('GRADE_PRE', u"""\\b([a-f]\\+?)\\s*-?\\s*grade\\b"""),

    ('THREED', u"""\\b3\\-?d\\b"""),

    ('DECADE', u"""\\b((?:18|19|20)?[0-9]0)(?:\\s*'\\s*)?s\\b"""),

    ('NUMBER', u"""(?<!\\w)([#\\p{Sc}+])?\\s?([0-9]+)([\\s,][0-9]{3})*(\\.[0-9]+)?\\s?(\\p{Sc}+|'?\\s?s)?(?!\\w)"""),

    ('HASHTAG', u"""\\#[a-zA-Z_][a-zA-Z0-9_]*"""),

    ('MENTION', u"""@[a-zA-Z0-9_]+"""),

    ('ASCIIARROW_R', u"""([\\-=]?\\>{2,}|[\\-=]+\\>)"""),  # -->, ==>, >>, >>>,

    ('ASCIIARROW_L', u"""(\\<{2,}[\\-=]?|\\<[\\-=]+)"""),  # <<<, <<, <==, <--

    ('MNDASH', u"""\\s\\-\\s|\\-{2,3}|\\u2013|\\u2014"""),

    ('ABBREV1', u"""\\b\\p{L}([&\\/\\-\\+])\\p{L}\\b"""),  # entities like S&P, w/o

    ('ABBREV2', u"""\\b(\\p{L})-(\\p{L}{2,})\\b"""),       # X-Factor, X-men, T-trak

    ('ABBREV3', u"""\\b(?:\\p{L}\\.){2,}"""),              # abbreviation with periods like U.S.

    ('ELLIPSIS', u"""(?:\\.\\s*){2,}|\\u2026"""),          # ellipsis

    ('XOXO', u"""\\b(?:[xX][oO])+\\b"""),

    ('PUNKT', u"""::+(?!\\/\\/)|[!?¡¿]+|[,\\.]"""),        # punctuation

    ('ANYWORD', u"""\\w+""")                               # any non-zero sequence of letters
]

DEFAULT_FEATURES = zip(*DEFAULT_FEATURE_MAP)[0]
FEATURE_PATTERNS = dict(DEFAULT_FEATURE_MAP)


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

    Hashtag and user mention regexes are simplified versions of
    https://github.com/twitter/twitter-text/blob/master/js/pkg/twitter-text-1.9.4.js#L217
    https://github.com/twitter/twitter-text/blob/master/js/pkg/twitter-text-1.9.4.js#L224
    For hashtags, we generate a plain word token as well as one with the
    appropriate character prefixed.  Note that hashtag are supposed to be
    prefixed by space or by string boundary, however this doesn't capture
    user intent (users may concatenate hashtags together).

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
                 features=DEFAULT_FEATURES,
                 groupname_format=u"<%s>",
                 word_buffer_len=5,
                 flags=re.UNICODE | re.DOTALL | re.VERBOSE,
                 debug=False):

        # attributes equal to parameter names
        feature_pattern = u'\n|\n'.join(
            u"(?P<%s>%s)" % (feature, FEATURE_PATTERNS[feature])
            for feature in features
        )
        self.regex = re.compile(feature_pattern, flags)
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

    def handle_reply(self, match, *args):
        period = STRIP_SPACES(match.group() or '')
        if period:
            yield self.groupname_format % u'PUBLIC'
        yield self._group_name(match)

    handle_xoxo = _group_tag
    handle_asciiarrow_l = _group_tag
    handle_asciiarrow_r = _group_tag
    handle_timeofday = _group_tag
    handle_date = _group_tag
    handle_mention = _group_tag

    def handle_customtoken(self, match, *args):
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

    def handle_emotic_west_l(self, match, *args):
        mouth = match.group(match.lastindex + 1)[0]
        group_name = match.lastgroup[:-2]   # drop '_L' suffix
        if mouth in u")]pdPD":
            group_name = u'_'.join([group_name, 'HAPPY'])
        elif mouth in u"([cCsSlL":
            group_name = u'_'.join([group_name, 'SAD'])
        yield match.group()
        yield self.groupname_format % group_name

    def handle_emotic_west_r(self, match, *args):
        mouth = match.group(match.lastindex + 1)[-1]
        group_name = match.lastgroup[:-2]   # drop '_R' suffix
        if mouth in u"([":
            group_name = u'_'.join([group_name, 'HAPPY'])
        elif mouth in u")]dD":
            group_name = u'_'.join([group_name, 'SAD'])
        yield match.group()
        yield self.groupname_format % group_name

    def handle_emotic_west_l_misc(self, match, *args):
        mouth = match.group(match.lastindex + 1)[0]
        group_name = match.lastgroup[:-7]  # drop '_L_MISC' suffix
        if mouth in u"$":
            group_name = u'_'.join([group_name, 'HAPPY'])
        elif mouth in u"@\\/":
            group_name = u'_'.join([group_name, 'SAD'])
        yield match.group()
        yield self.groupname_format % group_name

    handle_emotic_west_r_misc = handle_emotic_west_l_misc

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

    def _emphasis_like(self, match, *args):
        yield self._group_name(match)
        yield match.group(match.lastindex + 1)

    def _simple_named_handler(self, match, *args):
        yield self._group_name(match)
        yield match.group()

    def handle_hashtag(self, match, *args):
        yield self._group_name(match)
        yield match.group()
        yield match.group()[1:]

    handle_emotic_east_sad = _simple_named_handler

    handle_emphasis_b = _emphasis_like
    handle_emphasis_u = _emphasis_like
    handle_mpaarating = wrapped_entity_handler
    handle_threed = wrapped_entity_handler

    def handle_decade(self, match, *args):
        yield STRIP_NONWORD(match.group()).lower()

    def handle_number(self, match, *args):
        prefix = STRIP_SPACES(match.group(match.lastindex + 1) or '')
        num = STRIP_NONWORD(match.group(match.lastindex + 2) or '')
        thousands = STRIP_NONWORD(match.group(match.lastindex + 3) or '')
        floating = STRIP_NONWORD(match.group(match.lastindex + 4) or '')
        suffix = STRIP_SPACES(match.group(match.lastindex + 5) or '')
        if prefix:
            yield prefix
            if prefix == u'#' and (not thousands) and (not floating):
                yield prefix + num
                if suffix:
                    yield suffix
                return
        elif suffix and suffix[-1] == u's':
            num_len = len(num)
            if (not thousands) and (not floating) and ((num_len == 4 and num[-1] == u'0' and num[0] in [u'1', '2']) or (num_len == 2 and num[-1] == u'0')):
                yield num + u's'
                yield self.groupname_format % u'DECADE'
                return
        yield self.groupname_format % u'NUM'
        yield self.groupname_format % (u'NUM_%s' % (len(num) + len(thousands)))
        if floating:
            yield self.groupname_format % u'FLOAT'
            yield self.groupname_format % (u'FLOAT_%s' % len(floating))
        if suffix:
            yield suffix

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
            yield STRIP_SPACES(extracted)

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
