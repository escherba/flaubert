import multiprocessing
import logging
from itertools import chain
from pymaptools.io import read_json_lines, PathArgumentParser
from gensim.models import word2vec
from flaubert.conf import CONFIG

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_sentence_iter(input_files):
    containers = (chain.from_iterable(read_json_lines(fname)) for fname in input_files)
    return chain(*containers)


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--sentences', type=str, metavar='FILE', nargs='+',
                        help='Input files')
    parser.add_argument('--output', type=str, required=True,
                        help='where to save the model to')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of workers to use (default: same as number of CPUs)')
    namespace = parser.parse_args(args)
    return namespace


def build_model(input_files, model_output=None, num_workers=1):
    if num_workers is None or num_workers < 0:
        raise ValueError("Invalid value specified num_workers=%d" % num_workers)

    # Note: training gensim's word2vec directly on an iterator causes a drop in
    # accuracy, which should be investigated and probably filed as an issue.
    sentences = list(get_sentence_iter(input_files))
    logging.info("Training word2vec model on %d sentences", len(sentences))

    # Initialize and train the model (this will take some time)
    logging.info("Training model...")
    model = word2vec.Word2Vec(
        sentences, workers=num_workers, **CONFIG['word2vec'])

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    if model_output:
        model.save(model_output)

    return model


def run(args):
    logging.info("Reading sentences from files: %s", args.sentences)
    build_model(args.sentences, args.output, args.workers)


if __name__ == "__main__":
    run(parse_args())
