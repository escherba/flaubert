import logging
import sys
import json
from gensim.models import word2vec
from pymaptools.io import PathArgumentParser, GzipFileType
from flaubert.conf import CONFIG


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--sentences', type=GzipFileType('r'), nargs='*',
                        default=[sys.stdin],
                        help="input file (JSON-encoded sentences)")
    parser.add_argument('--output', type=str, required=True,
                        help='where to write model to')
    namespace = parser.parse_args(args)
    return namespace


def read_sentences(args):
    for finput in args.sentences:
        for line in finput:
            review_sentences = json.loads(line)
            for sentence in review_sentences:
                yield sentence


def run(args):
    sentences = list(read_sentences(args))

    logging.info("Training model...")
    model = word2vec.Word2Vec(
        sentences, workers=-1, **CONFIG['word2vec'])

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    model.save(args.output)


if __name__ == "__main__":
    run(parse_args())
