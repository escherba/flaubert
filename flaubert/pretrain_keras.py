
'''
    We loop over words in a dataset, and for each word, we look at a context window around the word.
    We generate pairs of (pivot_word, other_word_from_same_context) with label 1,
    and pairs of (pivot_word, random_word) with label 0 (skip-gram method).

    We use the layer WordContextProduct to learn embeddings for the word couples,
    and compute a proximity score between the embeddings (= p(context|word)),
    trained with our positive and negative labels.

    We then use the weights computed by WordContextProduct to encode words
    and demonstrate that the geometry of the embedding space
    captures certain useful semantic properties.

    Read more about skip-gram in this particularly gnomic paper by Mikolov et al.:
        http://arxiv.org/pdf/1301.3781v3.pdf

    Note: you should run this on GPU, otherwise training will be quite slow.
    On a EC2 GPU instance, expect 3 hours per 10e6 comments (~10e8 words) per epoch with dim_proj=256.
    Should be much faster on a modern GPU.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python skipgram_word_embeddings.py

    Dataset: 5,845,908 Hacker News comments.
    Obtain the dataset at:
        https://mega.co.nz/#F!YohlwD7R!wec0yNO86SeaNGIYQBOR0A
        (HNCommentsAll.1perline.json.bz2)
'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import theano

from flaubert.pretrain import get_sentences
from flaubert.keras_prep import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.embeddings import WordContextProduct, Embedding
from six.moves import range

from pymaptools.io import PathArgumentParser, pickle_dump, pickle_load

max_features = 50000  # vocabulary size: top 50,000 most common words in data
skip_top = 100        # ignore top 100 most common words
nb_epoch = 2
dim_proj = 256        # embedding space dimension
negative_samples = 1.
window_size = 4
model_optimizer = 'rmsprop'
model_loss = 'mse'

load_model = False
load_tokenizer = False


class SimplePickle(object):

    @classmethod
    def load(cls, input_file):
        model = pickle_load(input_file)
        assert isinstance(model, cls)

    def dump(self, output_file):
        pickle_dump(self, output_file)


class BasicEmbeddingWrapper(SimplePickle):

    def __init__(self, word_index=None, norm_weights=None):
        self.word_index = word_index or {}
        self.norm_weights = norm_weights if len(norm_weights) > 0 else []
        self.reverse_word_index = {v: k for k, v in list(word_index.items())}

    def embed_word(self, w):
        i = self.word_index.get(w)
        if (not i) or (i < skip_top) or (i >= max_features):
            return None
        return self.norm_weights[i]

    def closest_to_point(self, point, nb_closest=10):
        proximities = np.dot(self.norm_weights, point)
        tups = list(enumerate(proximities))
        tups.sort(key=lambda x: x[1], reverse=True)
        return [(self.reverse_word_index.get(t[0]), t[1]) for t in tups[:nb_closest]]

    def closest_to_word(self, w, nb_closest=10):
        i = self.word_index.get(w)
        if (not i) or (i < skip_top) or (i >= max_features):
            return []
        return self.closest_to_point(self.norm_weights[i].T, nb_closest)


''' the resuls in comments below were for:
    5.8M HN comments
    max_features = 50000
    skip_top = 100
    nb_epoch = 2
    dim_proj = 256
    negative_samples = 1.
    window_size = 4
    optimizer = rmsprop
    loss = mse
    and frequency subsampling of factor 10e-5.
'''

words = [
    "article",  # post, story, hn, read, comments
    "3",  # 6, 4, 5, 2
    "two",  # three, few, several, each
    "great",  # love, nice, working, looking
    "data",  # information, memory, database
    "money",  # company, pay, customers, spend
    "years",  # ago, year, months, hours, week, days
    "android",  # ios, release, os, mobile, beta
    "javascript",  # js, css, compiler, library, jquery, ruby
    "look",  # looks, looking
    "business",  # industry, professional, customers
    "company",  # companies, startup, founders, startups
    "after",  # before, once, until
    "own",  # personal, our, having
    "us",  # united, country, american, tech, diversity, usa, china, sv
    "using",  # javascript, js, tools (lol)
    "here",  # hn, post, comments
]


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--output', type=str, required=True,
                        help='path to full model')
    parser.add_argument('--simple_model', type=str, required=True,
                        help='path to simple model')
    parser.add_argument('--sentences', type=str, metavar='FILE', nargs='+',
                        required=True, help='Input files')
    parser.add_argument('--no_sentences', action='store_true',
                        help='use docs instead of sentences')
    parser.add_argument('--limit', type=int, default=None,
                        help='how many sentences to limit input to')
    parser.add_argument('--field', type=str, default='review',
                        help='Field name (Default: review)')
    namespace = parser.parse_args()
    return namespace


def get_texts(args):
    for sent in get_sentences(args):
        yield u' '.join(sent).encode('utf-8')


def run(args):

    print("Fit tokenizer...")
    tokenizer = text.Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(get_texts(args))

    # training process
    if load_model:
        print('Load model...')
        model = pickle_load(args.output)
    else:
        print('Build model...')
        model = Sequential()
        model.add(WordContextProduct(max_features, proj_dim=dim_proj, init="uniform"))
        model.compile(loss=model_loss, optimizer=model_optimizer)

        sampling_table = sequence.make_sampling_table(max_features)

        for e in range(nb_epoch):
            print('-' * 40)
            print('Epoch', e)
            print('-' * 40)

            progbar = generic_utils.Progbar(tokenizer.document_count)
            samples_seen = 0
            losses = []

            for i, seq in enumerate(tokenizer.texts_to_sequences_generator(get_texts(args))):
                # get skipgram couples for one text in the dataset
                couples, labels = sequence.skipgrams(
                    seq, max_features, window_size=window_size,
                    negative_samples=negative_samples, sampling_table=sampling_table)
                if couples:
                    # one gradient update per sentence (one sentence = a few 1000s of word couples)
                    X = np.array(couples, dtype="int32")
                    loss = model.train(X, labels)
                    losses.append(loss)
                    if len(losses) % 100 == 0:
                        progbar.update(i, values=[("loss", np.mean(losses))])
                        losses = []
                    samples_seen += len(labels)
            print('Samples seen:', samples_seen)
        print("Training completed!")

        print("Saving model...")
        pickle_dump(model, args.output)

    print("It's test time!")

    # recover the embedding weights trained with skipgram:
    weights = model.layers[0].get_weights()[0]

    # we no longer need this
    del model

    weights[:skip_top] = np.zeros((skip_top, dim_proj))
    norm_weights = np_utils.normalize(weights)
    word_index = tokenizer.word_index

    simplified_model = BasicEmbeddingWrapper(
        norm_weights=norm_weights,
        word_index=word_index
    )

    for w in words:
        res = simplified_model.closest_to_word(w)
        print('====', w)
        for r in res:
            print(r)

    # save final model
    simplified_model.dump(args.simple_model)


if __name__ == "__main__":
    run(parse_args())
