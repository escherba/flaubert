"""
Vectorize to a simple format for testing an LSTM script
"""

import sys
import argparse
from itertools import chain
import cPickle as pickle
from sklearn.cross_validation import train_test_split
from pymaptools.vectorize import enumerator
from pymaptools.io import GzipFileType, read_json_lines
from flaubert.conf import CONFIG


def vectorize_sentences(input_iter):
    enum = enumerator()
    for obj in input_iter:
        yield ([enum[w] for w in chain(*obj['X'])], obj['Y'])


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=GzipFileType('r'), nargs='*',
                        default=[sys.stdin], help='Input file(s)')
    parser.add_argument('--output', type=GzipFileType('wb'), required=True,
                        help='Output file')
    namespace = parser.parse_args(args)
    return namespace


def run(args):
    data = list(vectorize_sentences(chain(*(read_json_lines(fn) for fn in args.input))))
    X, y = zip(*data)
    cfg = CONFIG['train']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg['test_size'], random_state=cfg['random_state'])
    pickle.dump((X_train, y_train), args.output)
    pickle.dump((X_test, y_test), args.output)


if __name__ == "__main__":
    run(parse_args())
