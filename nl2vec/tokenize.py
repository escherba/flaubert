# -*- coding: utf-8 -*-

import regex as re
from collections import deque
from pymaptools.inspect import get_object_attrs


DEFAULT_FEATURE_MAP = u"""
(?P<SPECIAL>\\b__([A-Za-z]+)__\\b)
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
(?P<ASCIIARROW_RIGHT>([\\-=]?\\>{2,}|[\\-=]+\\>))        # -->, ==>, >>, >>>
|
(?P<ASCIIARROW_LEFT>(\\<{2,}[\\-=]?|\\<[\\-=]+))         # <<<, <<, <==, <--
|
(?P<COLON>:+(?!\\/\\/))                                  # colon (unless a part of URI scheme)
|
(?P<EMPHMARK>[!?¡¿]+)                                    # emphasis characters
|
(?P<ELLIPSIS>\\.+\\s*\\.+|\\u2026)                       # ellipsis
|
(?P<ABBREV1>\\b\\p{L}([&\\/\\-\\+])\\p{L}\\b)            # entities like S&P, w/o
|
(?P<ABBREV2>\\b(\\p{L})-(\\p{L}{2,})\\b)                 # X-Factor, X-men, T-trak
|
(?P<ABBREV3>\\b(?:\\p{L}\\.){2,})                        # abbreviation with periods like U.S.
|
(\\p{L}+)                                                # any non-zero sequence of letters
"""

RE_DEFAULT_FEATURES = re.compile(DEFAULT_FEATURE_MAP, re.VERBOSE | re.UNICODE)


class RegexFeatureTokenizer(object):
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
                 groupname_format=u"<%s>",
                 regex=RE_DEFAULT_FEATURES,
                 word_buffer_len=5,
                 debug=False):

        # attributes equal to parameter names
        self.regex = regex
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

    def handle_special(self, match, *args):
        tag_name = match.group(match.lastindex + 1).upper()
        yield self.groupname_format % tag_name

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

    def handle_ellipsis(self, match, *args):
        if match.group() == u"\u2026":
            yield u"..."
        else:
            yield match.group()

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
