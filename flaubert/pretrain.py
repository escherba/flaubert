import multiprocessing
import logging
import os
from itertools import islice, chain
from pymaptools.io import read_json_lines, PathArgumentParser
from flaubert.conf import CONFIG

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def sentence_iter(document_iter, cfg):
    split_by_sentence = cfg['split_by_sentence']
    allowed_labels = cfg['doc2vec_labels']
    sentence_label = 'sentence' in allowed_labels
    document_label = 'document' in allowed_labels
    if split_by_sentence:
        for doc_idx, document in enumerate(document_iter):
            doc_data = []
            for sent_idx, sentence in enumerate(document):
                labels = []
                if sentence_label:
                    labels.append(u'SENT_%d_%d' % (doc_idx, sent_idx))
                if document_label:
                    labels.append(u'DOC_%d' % doc_idx)
                doc_data.append((sentence, labels))
            yield doc_data
    else:
        for doc_idx, document in enumerate(document_iter):
            labels = []
            if document_label:
                labels.append(u'DOC_%d' % doc_idx)
            yield [(list(chain.from_iterable(document)), labels)]


def doc_iter(args):
    field = args.field
    for fname in args.input:
        for doc in read_json_lines(fname):
            yield doc[field]


def get_sentences(args):
    logging.info("Reading sentences from files: %s", args.input)
    iterator = doc_iter(args)
    if args.limit:
        iterator = islice(iterator, args.limit)
    if CONFIG['pretrain']['split_by_sentence']:
        logging.info("Using documents split by sentences")
        for doc in iterator:
            for sentence in doc:
                yield sentence
    else:
        logging.info("Using whole documents")
        for doc in iterator:
            yield list(chain.from_iterable(doc))


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--input', type=str, metavar='FILE', nargs='+',
                        help='Input files')
    parser.add_argument('--field', type=str, default='review',
                        help='Field name (Default: review)')
    parser.add_argument('--verbose', action='store_true',
                        help='be verbose')
    parser.add_argument('--output', type=str, required=True,
                        help='where to save the model to')
    parser.add_argument('--limit', type=int, default=None,
                        help='(for debugging) limit input to n lines')
    parser.add_argument('--corpus_model', type=str, default=None,
                        help='where corpus model lives (GloVe)')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of workers to use (default: same as number of CPUs)')
    parser.add_argument('--doc2vec', action='store_true',
                        help='use Doc2Vec instead of Word2Vec model')
    namespace = parser.parse_args(args)
    return namespace


def build_model_word2vec(args, replace_sims=True):

    from gensim.models import word2vec, doc2vec

    def get_labeled_sentences(args):
        logging.info("Reading sentences+labels from files: %s", args.input)
        for doc_data in sentence_iter(doc_iter(args), cfg=CONFIG['pretrain']):
            for sentence, labels in doc_data:
                yield doc2vec.LabeledSentence(sentence, labels=labels)

    num_workers = args.workers
    if num_workers is None or num_workers < 0:
        raise ValueError("Invalid value specified num_workers=%d" % num_workers)

    # TODO: training gensim's word2vec directly on an iterator causes a drop
    # in accuracy, which should be investigated and probably filed as an issue.

    # Initialize and train the model (this will take some time)
    embedding_type = CONFIG['pretrain']['embedding']
    if embedding_type == 'doc2vec':
        sentences = list(get_labeled_sentences(args))
        logging.info("Training doc2vec model on %d sentences", len(sentences))
        model = doc2vec.Doc2Vec(
            sentences, workers=num_workers, **CONFIG['doc2vec'])
    elif embedding_type == 'word2vec':
        sentences = list(get_sentences(args))
        logging.info("Training word2vec model on %d sentences", len(sentences))
        model = word2vec.Word2Vec(
            sentences, workers=num_workers, **CONFIG['word2vec'])
    else:
        raise ValueError("Invalid config setting embedding=%s" % embedding_type)

    if replace_sims:
        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)

    return model


def build_model_glove(args):

    from glove import Glove, Corpus

    if not os.path.exists(args.corpus_model) or \
            max(map(os.path.getmtime, args.input)) >= os.path.getmtime(args.corpus_model):

        # Build the corpus dictionary and the cooccurrence matrix.
        logging.info('Pre-processing corpus')

        corpus_model = Corpus()
        corpus_model.fit(get_sentences(args), window=CONFIG['glove']['window'])
        corpus_model.save(args.corpus_model)

        logging.info('Dict size: %s' % len(corpus_model.dictionary))
        logging.info('Collocations: %s' % corpus_model.matrix.nnz)
    else:
        # Try to load a corpus from disk.
        logging.info('Reading corpus statistics')
        corpus_model = Corpus.load(args.corpus_model)

        logging.info('Dict size: %s' % len(corpus_model.dictionary))
        logging.info('Collocations: %s' % corpus_model.matrix.nnz)

    # Train the GloVe model and save it to disk.
    logging.info('Training the GloVe model')

    glove = Glove(no_components=CONFIG['glove']['size'], learning_rate=CONFIG['glove']['learning_rate'])
    glove.fit(corpus_model.matrix, epochs=CONFIG['glove']['epochs'],
              no_threads=args.workers, verbose=args.verbose)
    glove.add_dictionary(corpus_model.dictionary)
    return glove


def run(args):

    if CONFIG['pretrain']['algorithm'] == 'glove':
        model = build_model_glove(args)
    elif CONFIG['pretrain']['algorithm'] == 'word2vec':
        model = build_model_word2vec(args)
    else:
        raise ValueError("Invalid algorithm %s" % CONFIG['pretrain']['algorithm'])

    if args.output:
        model.save(args.output)


if __name__ == "__main__":
    run(parse_args())
