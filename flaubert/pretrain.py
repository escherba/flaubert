import logging
import sys
import json
from gensim.models import word2vec
from pymaptools.io import PathArgumentParser, GzipFileType


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
    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    sentences = list(read_sentences(args))

    # Initialize and train the model (this will take some time)
    print "Training model..."
    model = word2vec.Word2Vec(
        sentences, workers=num_workers, size=num_features,
        min_count=min_word_count, window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)


if __name__ == "__main__":
    run(parse_args())
