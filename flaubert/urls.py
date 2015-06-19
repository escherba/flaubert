# -*- coding: utf-8 -*-

import rfc3987
import regex as re
from functools import partial
from pkg_resources import resource_filename
from pymaptools.io import read_text_resource
from pymaptools.iter import roundrobin
from pymaptools.types import Struct


_resource = partial(resource_filename, __name__)


class URLRegexBuilder(object):
    def __init__(self, uri_require_scheme=False, uri_valid_tlds=False, uri_specific_tlds=True,
                 email_require_scheme=False, email_valid_tlds=True):
        """Use three patterns to detect emails, URIs with scheme, and URIs w/o scheme

        To detect URIs with scheme, it is not necessary to ensure that the TLD suffix is valid
        since the scheme plus hierarchical path pattern is quite specific, and will capture user
        intent (to post a URL) even if the posted URL is incorrect.

        A general rule of this kind will not work for those URIs that don't have scheme, however.
        Here, unless we have a list of known/valid TLDs, we get far too many false positives
        (pretty much any two words separated by a dot will register as a URL). A list of all valid
        TLD suffixes reduces false positive rate, but is still not enough. For good matches, we
        want *specific* TLDs (either very popular TLDs or those that don't look like a common word).

        Some references used when creating this:
            https://github.com/dgerber/rfc3987/blob/master/rfc3987.py
            https://github.com/twitter/twitter-text/blob/master/java/src/com/twitter/Regex.java
            http://daringfireball.net/2010/07/improved_regex_for_matching_urls
            https://mathiasbynens.be/demo/url-regex
            http://www.regexguru.com/2008/11/detecting-urls-in-a-block-of-text/
        """

        valid_latin_tld = u"[a-z]{2,}"
        valid_punycode = u"xn\\-\\-[a-z0-9]+"

        # all possible TLDs, including invalid ones
        nonspecific_tlds = valid_latin_tld + u'|' + valid_punycode

        # all valid TLDs
        valid_tlds = \
            u'|'.join(read_text_resource(_resource('tlds-alpha-by-domain.txt')))

        # most common TLDs from the valid ones, plus unspecified XN-- ones
        specific_tlds = \
            u'|'.join(read_text_resource(_resource('tlds-specific.txt'))) + \
            u'|' + valid_punycode

        # http, https, and ftp schemes only (sorry, Gopher)
        # TODO: make this work with internationalized URLs (Arabic, etc)
        scheme_template = u"(?P<scheme>https?:\\/\\/|ftp:\\/\\/)"
        host_component = u"""
            [a-z0-9]            # first char of domain component, no hyphen
            [a-z0-9-]*          # middle of domain component
            (?<!-)              # last char cannot be a hyphen
        """
        hspace = u"[^\\S\\n]"               # double negative for any space except a new line
        authority_template = u"""
            (?P<authority>                  # authority begins with word break (required)
                (?:%%(userinfo)s@)?         # Optional user name and password (sep. by colon)
                (?:
                    (?<host>                # begin host
                        \\b(?:%(host_component)s\\.)+(?:%(tld)s)\\b
                        |
                        www%(hspace)s*\\.%(hspace)s*
                        (?:%(host_component)s%(hspace)s*\\.%(hspace)s*)+
                        (?:com|net|info|biz|co\\.uk)
                    )
                    |                       # or
                    (?<ip_addr>             # IP address
                        \\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b
                    )
                )                           # end host (non-optional)
                (?::%%(port)s)?             # optional port (filled by rfc3987)
            )                               # end authority (authority component is non-optional)
        """

        # unlike using rfc3987 characters to fill in path group, this expression
        # allows matching parentheses (up to two levels)
        path_template = u"""
            (?P<path>\\/                    # A path begins with slash (required)
                (?:                         # One or more:
                    [^'\\"\\?\\#\\s<>]+     # Run of non-space, non-<>, non-quote,
                )*                          # non-pound sign, non-question mark
            )?                              # end path
        """

        # when mathching with scheme present, we can be more liberal with TLDs,
        # otherwise it is better to use specific TLDs for more strictness.
        noscheme_prefix = authority_template % dict(
            tld=(specific_tlds if uri_specific_tlds else nonspecific_tlds),
            host_component=host_component,
            hspace=hspace
        )
        scheme_prefix = scheme_template + authority_template % dict(
            tld=(valid_tlds if uri_valid_tlds else nonspecific_tlds),
            host_component=host_component,
            hspace=hspace
        )

        # first part of the URI is different depending on whether we're matching
        # a URL or an email
        uri_first_part = scheme_prefix if uri_require_scheme \
            else u'(?:{})'.format(u'|'.join([scheme_prefix, noscheme_prefix]))

        # second part of the URI is the same for both URLs and emails
        uri_second_part = u"""
            (?P<path_component>
                %(path_template)s
                (?:\\?%%(query)s)?      # optional query (filled in by rfc3987 allowed chars)
                (?:\\#%%(fragment)s)?   # optional anchor (filled in by rfc3987 allowed chars)
            )
        """ % dict(
            path_template=path_template,
        )

        # Finally,
        #
        # 1. Create a URI regex
        self.uri_regex = u"""
        (?P<uri>
            %(uri_first_part)s
            %(uri_second_part)s
        )
        """ % dict(
            uri_first_part=uri_first_part,
            uri_second_part=uri_second_part
        )

        # 2. Create an Email regex
        #
        # for email, we are liberal with scheme but relatively strict with TLDs;
        # and also strict with userinfo (don't allow password or very strange
        # usernames)
        self.email_regex = u"""
        (?P<email>
            (?P<scheme>mailto:|e\\-?mail:|message:|via:)%(scheme_opt)s  # scheme
            (?P<authority>\\b
                (?P<userinfo>\\w+(?:[\\-\\+\\.']\\w+)*)@
                (?P<host>               # begin host
                    (?:%(host_component)s\\.)+
                    (?:%(tld)s)         # top-level domain (important: does not include XN-- tlds)
                )                       # end host (non-optional)
                \\b                     # required word break
            )
            %(uri_second_part)s
        )
        """ % dict(
            scheme_opt=u'' if email_require_scheme else u'?',
            host_component=host_component,
            tld=valid_tlds if email_valid_tlds else nonspecific_tlds,
            uri_second_part=uri_second_part
        )

    def build(self, *which):
        """
        :param which: a (sub)set of ["uri", "email]
        """
        regex_components = []
        for regex_type in which:
            regex_components.append(getattr(self, regex_type + '_regex'))
        string = u'(?P<string>\n%s\n)' % u'\n|\n'.join(regex_components)
        return rfc3987.get_compiled_pattern(string, re.IGNORECASE | re.VERBOSE | re.UNICODE)


class URI(Struct):

    readwrite_attrs = frozenset(
        ["email", "uri", "ip_addr", "scheme", "userinfo", "host", "port", "path", "query",
         "fragment", "authority", "meta", "domain_and_suffix", "domain", "suffix",
         "subdomain", "entity_type", "string", "path_component"])

    @classmethod
    def fromDict(cls, data):
        return super(URI, cls).fromDict(data)

    @classmethod
    def fromMatch(cls, match):
        """Convert a URI match to email if user info is present and scheme is absent

        Note that schemes could be different for Emails since people will type things like
        email:someone@example.com, via:someone@example.com as well as the standard
        mailto:someone@example.com

        Also note that sometimes people write their emails like this:
        EMAIL.DR.ABEGBESPELLHOME@HOTMAIL.COM so we strip off the "email" prefix if
        it exists followed by non-word characters
        """
        data = match.groupdict()
        obj = cls.fromDict(data)
        uri = data.get('uri')
        email = data.get('email')
        if uri and not email:
            obj.entity_type = 'uri'
        elif email and not uri:
            obj.entity_type = 'email'
        else:
            raise ValueError(u"Ambiguous match %s (both `uri` and `email` matched)" % data)
        return obj


class URLParser(object):
    """
    URL finder and normalizer

    also see IETF spec regex:
    http://www.ietf.org/rfc/rfc3986.txt

    for list of valid TLDs, see:
    http://data.iana.org/TLD/tlds-alpha-by-domain.txt

    For examples of similar patterns:
    http://daringfireball.net/2010/07/improved_regex_for_matching_urls
    https://gist.github.com/uogbuji/705383
    https://gist.github.com/gruber/249502
    http://alanstorm.com/url_regex_explained
    http://mathiasbynens.be/demo/url-regex

    Spammers often take advantage of those shorteners for which they can
    obtain multiple unique URLs pointing to the same piece of content.
    Note that there are two lists of URL shorteners this class reads.
    The "whitelist" isn't actually a whitelist in the sense that content
    with domains occuring in the list will be automatically accepted.
    Domains added to "whitelist" are either (a) shorteners owned by
    established newspapers or magazines, which means the content is
    hosted by those networks and is likely to be high-quality, or
    (b) social sites like Vine or YouTube (?) for which it can be said
    that each piece of hosted content corresponds to only one link and
    are therefore difficult to be taken advantage of by spammers.
    The "custom" list of shortener domains contains all other shorteners.
    """

    # skip these characters at the very end
    RE_URL_TRAILING = partial(
        re.compile(u"""[\\p{Z}\\p{InGeometric_Shapes}\\p{InDingbats}\\p{Pd}`\\!\\[\\]{};:'\".,<>\\?«»“”‘’]+$""").subn,
        u'')

    RE_PREFORMAT = partial(re.compile(u'\\|+').sub, u'')

    def __init__(self, encoding='utf-8', url_token="__URL__",
                 normalize_shortener_urls=True, unique_urls=False,
                 sort_urls=False, prepend_urls=False, replace_urls=False,
                 overlapped_urls=False,
                 email_require_scheme=False, email_valid_tlds=True,
                 uri_require_scheme=False, uri_valid_tlds=False, uri_specific_tlds=True,
                 uri_types="email,uri", placeholder_format=None):
        """
        For an accessible intro to URL syntax see:
            http://en.wikipedia.org/wiki/URI_scheme#Generic_syntax
        """
        uri_types = [t.strip() for t in uri_types.split(',')]
        self._re_url_iter = URLRegexBuilder(
            uri_require_scheme=uri_require_scheme,
            uri_valid_tlds=uri_valid_tlds,
            uri_specific_tlds=uri_specific_tlds,
            email_require_scheme=email_require_scheme,
            email_valid_tlds=email_valid_tlds).build(*uri_types).finditer
        self._encoding = encoding
        self._sort_urls = sort_urls  # only available in prepend_urls mode
        self._prepend_urls = prepend_urls
        self._unique_urls = unique_urls
        self._replace_urls = replace_urls
        self._overlapped_urls = overlapped_urls
        self._placeholder_format = placeholder_format
        self._normalize_shortener_urls = normalize_shortener_urls
        self._url_token = u" {} ".format(url_token.strip())

    def parse_urls(self, text, overlapped=True):
        """
        :param text: content to process
        :type text: unicode
        :param url_token: token to insert in place of URL
        :type url_token: str, unicode
        :return: a tuple of content stripped of URLs and array of
                 URLComponents
        :rtype: tuple

        Detect run-ons. A run-on URL is one that would normally be recognized
        as a single URL but that is really two urls:

        >>> norm = URLParser()
        >>> len(norm.parse_urls("A quick brown fox http://t.co/AF2EDS7 -->")[1])
        1
        >>> len(norm.parse_urls('A quick <a href="http://t.co/AF2EDS7">fox</a> -->')[1])
        1
        >>> len(norm.parse_urls('')[1])
        0
        >>> text = u"I just got up to 12,584 in #DoodleJump!!! Beat that! http://t.co/X2HYxZBLCd\\n"
        >>> cf, urls = norm.parse_urls(text)
        >>> cf
        [u'I just got up to 12,584 in #DoodleJump!!! Beat that! ', u'\\n']
        """
        urls = []
        content_fragments = []
        end = 0
        prev_end = 0
        placeholder_format = self._placeholder_format
        for match in self._re_url_iter(text, overlapped=overlapped):
            matched_string, num_trailing = self.RE_URL_TRAILING(match.group())
            start, cursor = match.span()
            if num_trailing > 0:
                cursor -= num_trailing
                match = self._re_url_iter(matched_string).next()
            # strip closing parenthesis if no opening found
            if matched_string[-1] == u')' and u'(' not in matched_string:
                cursor -= 1
                match = self._re_url_iter(matched_string[0:-1]).next()
            cursor = max(end, cursor)
            url = URI.fromMatch(match)
            if end < start:
                content_fragments.append(text[end:start])
            end = cursor
            # check for run-on URLs if have scheme and previous URLs exist
            if urls and overlapped:
                prev_url = urls[-1]
                prev_string = unicode(prev_url)
                curr_string = unicode(url)
                if url.scheme and start < prev_end and match.groupdict()['scheme']:
                    # at this point, the match must have a scheme, otherwise
                    # we will match URLs with scheme multiple times due to
                    # overlap
                    split_prev = prev_string.split(curr_string)
                    num_occurences = len(split_prev) - 1
                    if num_occurences > 0:
                        # run-on URL encountered, attempt to correct previous URL
                        urls.pop(-1)
                        for inner_match in self._re_url_iter(u''.join(split_prev), overlapped=False):
                            new_url = URI.fromMatch(inner_match)
                            urls.append(new_url)
                            if placeholder_format:
                                content_fragments.append(placeholder_format % new_url.entity_type)
                    for _ in xrange(num_occurences):
                        urls.append(url)
                        if placeholder_format:
                            content_fragments.append(placeholder_format % url.entity_type)
                elif start >= prev_end:
                    urls.append(url)
                    if placeholder_format:
                        content_fragments.append(placeholder_format % url.entity_type)
            else:
                urls.append(url)
                if placeholder_format:
                    content_fragments.append(placeholder_format % url.entity_type)

            prev_end = end
        content_fragments.append(text[end:])
        return content_fragments, urls

    def parse_url(self, text):
        _, urls = self.parse_urls(text)
        if urls:
            return urls[0]
        else:
            return None

    def find_urls(self, text, overlapped=True):
        return self.parse_urls(text, overlapped=overlapped)[1]

    def whiteout_urls(self, text, token_format=u" __%s__ "):
        final_els = []
        for element in roundrobin(*self.parse_urls(text)):
            if isinstance(element, URI):
                final_els.append(token_format % element.entity_type)
            else:
                final_els.append(element)
        return u''.join(final_els)
