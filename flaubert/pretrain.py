import multiprocessing
import logging
from itertools import chain
from pymaptools.io import read_json_lines, PathArgumentParser
from gensim.models import word2vec, doc2vec
from flaubert.conf import CONFIG

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_sentences(input_files):
    containers = (chain.from_iterable(read_json_lines(fname)) for fname in input_files)
    return chain(*containers)


def sentence_iter(input_files):
    doc_idx = 0
    allowed_labels = CONFIG['doc2vec_labels']
    sentence_label = 'sentence' in allowed_labels
    document_label = 'document' in allowed_labels
    for fname in input_files:
        for document in read_json_lines(fname):
            doc_data = []
            for sent_idx, sentence in enumerate(document):
                labels = []
                if sentence_label:
                    labels.append(u'SENT_%d_%d' % (doc_idx, sent_idx))
                if document_label:
                    labels.append(u'DOC_%d' % doc_idx)
                doc_data.append((sentence, labels))
            yield doc_data
            doc_idx += 1


def get_labeled_sentences(input_files):
    for doc_data in sentence_iter(input_files):
        for sentence, labels in doc_data:
            yield doc2vec.LabeledSentence(sentence, labels=labels)


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--sentences', type=str, metavar='FILE', nargs='+',
                        help='Input files')
    parser.add_argument('--output', type=str, required=True,
                        help='where to save the model to')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of workers to use (default: same as number of CPUs)')
    parser.add_argument('--doc2vec', action='store_true',
                        help='use Doc2Vec instead of Word2Vec model')
    namespace = parser.parse_args(args)
    return namespace


def build_model(args):

    input_files = args.sentences
    num_workers = args.workers

    if num_workers is None or num_workers < 0:
        raise ValueError("Invalid value specified num_workers=%d" % num_workers)

    # TODO: training gensim's word2vec directly on an iterator causes a drop
    # in accuracy, which should be investigated and probably filed as an issue.

    logging.info("Reading sentences from files: %s (%d workers)", input_files, num_workers)

    # Initialize and train the model (this will take some time)
    if CONFIG['embedding'] == 'doc2vec':
        sentences = list(get_labeled_sentences(input_files))
        logging.info("Training doc2vec model on %d sentences", len(sentences))
        model = doc2vec.Doc2Vec(
            sentences, workers=num_workers, **CONFIG['doc2vec'])
    elif CONFIG['embedding'] == 'word2vec':
        sentences = list(get_sentences(input_files))
        logging.info("Training word2vec model on %d sentences", len(sentences))
        model = word2vec.Word2Vec(
            sentences, workers=num_workers, **CONFIG['word2vec'])
    else:
        raise ValueError("Invalid config setting embedding=%s" % CONFIG['embedding'])

    return model


def run(args):

    model = build_model(args)
    model_output = args.output

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    if model_output:
        model.save(model_output)


if __name__ == "__main__":
    run(parse_args())
