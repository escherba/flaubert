# -*- coding: utf-8 -*-

import regex as re
from collections import deque
from pymaptools.inspect import get_object_attrs


DEFAULT_FEATURE_MAP = u"""
(?P<EMOTICON_EASTERN_LOW>\\(?[\\+\\^ˇ\\*\\->~][_\\.][\\+\\^ˇ\\*\\-<~]\\)?)
|
(?P<EMOTICON_EASTERN_HIGH>\\(?[\\^ˇ\\*][\\-~oO][\\^ˇ\\*]\\)?)
|
(?P<EMOTICON_WESTERN_LEFT>\\>?(?:=|(?:[:;]|(?<![\\w\\(\\)])[Bb])[\\-=\\^']?)(?:[\\(\\)\\*\\[\\]\\|]+|[cCoOpPdD]+\\b))
|
(?P<EMOTICON_WESTERN_RIGHT>(?<!\\w)[\\(\\)\\[\\]\\|]+(?:(?:[\\-=\\^'][:;]?|[\\-=\\^']?[:;])(?![\\w\\(\\)])|=))
|
(?P<EMOTICON_WESTERN_LEFT_ALT>(?<![0-9])[:;]3+\\b)
|
(?P<EMOTICON_RUSSIAN_HAPPY>\\){2,})
|
(?P<EMOTICON_RUSSIAN_SAD>\\({2,})
|
(?P<EMOTICON_HEART>(?<![0-9])\\<\\/?3+\\b)
|
(?P<ASCIIARROW_RIGHT>([\\-=]?\\>{2,}|[\\-=]+\\>))         # -->, ==>, >>, >>>
|
(?P<ASCIIARROW_LEFT>(\\<{2,}[\\-=]?|\\<[\\-=]+))          # <<<, <<, <==, <--
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
                 regex=RE_DEFAULT_FEATURES,
                 word_buffer_len=5,
                 debug=False):

        # attributes equal to parameter names
        self.regex = regex
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
