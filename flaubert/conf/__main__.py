import argparse
from pymaptools import nested_get
from flaubert.conf import CONFIG


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, nargs='+', required=True,
                        help='which nested key to lookup')
    namespace = parser.parse_args(args)
    return namespace


def run(args):
    print nested_get(CONFIG, args.key)


if __name__ == "__main__":
    run(parse_args())
