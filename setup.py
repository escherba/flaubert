import re
import itertools
from setuptools import setup, find_packages
from pkg_resources import resource_string
from glob import glob


# dependency links
SKIP_RE = re.compile(r'^\s*(?:-\S+)\s+(.*)$')

# Regex groups: 0: URL part, 1: package name, 2: package version
EGG_RE = re.compile(r'^(.+)#egg=([a-z0-9_.]+)(?:-([a-z0-9_.-]+))?$')

# Regex groups: 0: URL part, 1: package name, 2: branch name
URL_RE = re.compile(r'^\s*(https?://[\w\.]+.*/([^\/]+)/archive/)([^\/]+).zip$')

# our custom way of specifying extra requirements in separate text files
EXTRAS_RE = re.compile(r'.*\bextras\.(\w+)\.txt$')


def parse_reqs(reqs):
    """Parse requirements.txt files into lists of requirements and dependencies
    """
    pkg_reqs = []
    dep_links = []
    for req in reqs:
        # find things like `--find-links <URL>`
        dep_link_info = SKIP_RE.match(req)
        if dep_link_info is not None:
            url = dep_link_info.group(1)
            dep_links.append(url)
            continue
        # add packages of form:
        # git+https://github.com/Livefyre/pymaptools#egg=pymaptools-0.0.3
        egg_info = EGG_RE.match(req)
        if egg_info is not None:
            url, egg, version = egg_info.group(0, 2, 3)
            #if version is None:
            #    pkg_reqs.append(egg)
            #else:
            #    pkg_reqs.append(egg + '==' + version)
            dep_links.append(url)
            continue
        # add packages of form:
        # https://github.com/escherba/matplotlib/archive/qs_fix_build.zip
        zip_info = URL_RE.match(req)
        if zip_info is not None:
            url, pkg = zip_info.group(0, 2)
            pkg_reqs.append(pkg)
            dep_links.append(url)
            continue
        pkg_reqs.append(req)
    return pkg_reqs, dep_links


def build_extras(glob_pattern):
    """Generate extras_require mapping
    """
    fnames = glob(glob_pattern)
    result = dict()
    dep_links = []
    for fname in fnames:
        extras_match = EXTRAS_RE.match(fname)
        if extras_match is not None:
            extras_file = extras_match.group(0)
            extras_name = extras_match.group(1)
            with open(extras_file, 'r') as fhandle:
                result[extras_name], deps = parse_reqs(fhandle.readlines())
                dep_links.extend(deps)
    return result, dep_links


INSTALL_REQUIRES, INSTALL_DEPS = parse_reqs(
    resource_string(__name__, 'requirements.txt').splitlines())
TESTS_REQUIRE, TESTS_DEPS = parse_reqs(
    resource_string(__name__, 'requirements-tests.txt').splitlines())
EXTRAS_REQUIRE, EXTRAS_DEPS = build_extras('requirements-extras.*.txt')
DEPENDENCY_LINKS = list(set(itertools.chain(
    INSTALL_DEPS,
    TESTS_DEPS,
    EXTRAS_DEPS
)))

setup(
    name="spam-svc",
    version='0.0.19.post0',
    author="Eugene Scherba",
    author_email="escherba@gmail.com",
    description=("Tools for word2vec-based sentiment analysis"),
    url='https://github.com/escherba/nl2vec',
    packages=find_packages(exclude=['tests']),
    long_description=resource_string(__name__, 'README.rst'),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=TESTS_REQUIRE,
    dependency_links=DEPENDENCY_LINKS,
    test_suite='nose.collector',
    classifiers=[
    ]
)
