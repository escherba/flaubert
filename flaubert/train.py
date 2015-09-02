from __future__ import print_function

import numpy as np
import logging
import cPickle as pickle
from itertools import chain, izip
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from pymaptools.io import PathArgumentParser, GzipFileType, read_json_lines, open_gz
from flaubert.pretrain import sentence_iter
from flaubert.utils import ItemSelector, pd_row_iter, BagVectorizer
from flaubert.conf import CONFIG

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class GloveWrapper(object):

    """
    mixin to make a glove-python model look superficially like gensim.word2vec model
    """

    @property
    def syn0(self):
        return self.word_vectors

    def __getitem__(self, word):
        try:
            return self.word_vectors[self.dictionary[word]]
        except KeyError:
            return None


if CONFIG['train']['nltk_stop_words']:
    STOP_WORDS = frozenset(stopwords.words(CONFIG['train']['nltk_stop_words']))
else:
    STOP_WORDS = frozenset([])


def makeFeatureVec(words, model, num_features):
    """
    average all of the word vectors in a given paragraph
    """
    # Pre-initialize an empty numpy array (for speed)
    vector = np.zeros((num_features,), dtype="float32")
    nwords = 0
    for word in words:
        if word in STOP_WORDS:
            continue
        try:
            word_vector = model[word]
        except KeyError:
            continue
        if word_vector is not None:
            vector = np.add(vector, word_vector)
            nwords = nwords + 1
    # Divide the result by the number of words to get the average
    vector = np.divide(vector, float(nwords))
    return vector


def get_word2vec_features(document_iter, model):
    """
    Given a set of reviews (each one a list of words), calculate
    the average feature vector for each one and return a 2D numpy array
    """
    _, num_features = model.syn0.shape

    reviews = list(document_iter)
    num_reviews = len(reviews)

    # Preallocate a 2D numpy array, for speed
    result = np.zeros((num_reviews, num_features), dtype="float32")
    for idx, doc in enumerate(reviews):
        result[idx] = makeFeatureVec(chain(*doc), model, num_features)
    return result


def get_doc2vec_features(document_iter, model):

    _, num_features = model.syn0.shape

    doc_sent_labels = list(sentence_iter(document_iter))
    num_docs = len(doc_sent_labels)

    # Preallocate a 2D numpy array, for speed
    result = np.zeros((num_docs, num_features), dtype="float32")

    for idx, labels in enumerate(doc_sent_labels):
        nvecs = 0
        vector = np.zeros((num_features,), dtype="float32")
        for sentence, this_labels in labels:
            this_vector = makeFeatureVec(chain(sentence, this_labels), model, num_features)
            vector = np.add(vector, this_vector)
            nvecs += 1
        vector = np.divide(vector, float(nvecs))
        result[idx] = vector

    return result


def get_bow_features(documents):
    # TODO: replace this with a Pipeline
    data = [Counter(w for w in chain(*doc) if w not in STOP_WORDS) for doc in documents]
    vectorizer = DictVectorizer()
    train_data_features = vectorizer.fit_transform(data)
    transformer = TfidfTransformer()
    train_data_features = transformer.fit_transform(train_data_features)
    return train_data_features


def get_mixed_features(sentences, embedding_vectors, y_labels):
    X_tmp = []
    y = []
    for doc, emb_vec, y_label in izip(sentences, embedding_vectors, y_labels):
        if np.isnan(emb_vec).any():
            continue
        bow_feats = Counter(chain(*doc))
        y.append(y_label)
        X_tmp.append((bow_feats, emb_vec))

    num_rows = len(X_tmp)
    num_cols = len(X_tmp[0][1])
    X = np.recarray(
        shape=(num_rows,),
        dtype=[('bow', object), ('embedding', np.float32, (num_cols,))]
    )
    for i, (bow, embedding) in enumerate(X_tmp):
        X['bow'][i] = bow
        X['embedding'][i] = embedding
    return X, np.asarray(y)


PARAM_GRIDS = {
    'lr': [
        LogisticRegression,
        [
            {'clf__dual': [False], 'clf__penalty':['l1', 'l2'], 'clf__C': [0.01, 0.033, 0.1, 0.33, 1.0]},
            {'clf__dual': [True],  'clf__penalty':['l2'],       'clf__C': [0.01, 0.033, 0.1, 0.33, 1.0]}
        ]
    ],
    'svm': [
        LinearSVC,
        [
            {'clf__dual': [False], 'clf__penalty':['l1', 'l2'], 'clf__C': [1, 3.33, 10, 33, 100, 333]},
            {'clf__dual': [True],  'clf__penalty':['l2'],       'clf__C': [0.1, 1, 10, 100]}
        ]
    ],
    'random_forest': [
        RandomForestClassifier,
        {
            "clf__n_estimators": [90],
            "clf__max_depth": [32, 64],
            "clf__max_features": [50, 75, 100],
            "clf__min_samples_split": [2],
            "clf__min_samples_leaf": [2, 3],
            "clf__bootstrap": [False],
            "clf__criterion": ["gini"]
        }
    ],
    'adaboost': [
        (
            lambda: AdaBoostClassifier(DecisionTreeClassifier(
                criterion="gini", max_depth=2, min_samples_split=2, min_samples_leaf=2))
        ),
        {
            'clf__n_estimators': [60],
            'clf__learning_rate': [0.8],
            'clf__algorithm': ['SAMME.R']
        }
    ]
}


def build_grid(args, is_mixed):
    clf, clf_params = PARAM_GRIDS[CONFIG['train']['classifier']]
    feature_set_names = CONFIG['train']['features']
    if is_mixed:
        transformer_list = []
        transformer_weights = {}
        if set(feature_set_names).intersection(['word2vec', 'doc2vec']):
            transformer_weights['embedding'] = 1.0
            transformer_list.append(
                # ('embedding', Pipeline([
                #     ('selector', ItemSelector(key='embedding')),
                # ])),
                ('embedding', ItemSelector(key='embedding'))
            )
        if set(feature_set_names).intersection(['bow']):
            transformer_weights['bow'] = 1.0
            transformer_list.append(
                ('bow', Pipeline([
                    ('selector', ItemSelector(key='bow')),
                    ('vect', DictVectorizer()),
                    ('tfidf', TfidfTransformer()),
                ]))
            )
        pipeline = Pipeline([
            ('union', FeatureUnion(
                transformer_list=transformer_list,
                transformer_weights=transformer_weights,
            )),
            ('clf', clf()),
        ])
    else:
        pipeline = Pipeline([
            ('clf', clf()),
        ])
    grid_args = [pipeline, clf_params]
    grid_kwargs = dict(cv=5, scoring=CONFIG['train']['scoring'], n_jobs=-1, verbose=10)
    result = GridSearchCV(*grid_args, **grid_kwargs)
    return result


def drop_nans(X, y):
    X_nans = np.isnan(X).any(axis=1)
    y_nans = np.asarray(np.isnan(y))
    nans = X_nans | y_nans
    y = y[~nans]
    X = X[~nans]
    return X, y


def train_model(args, X_train, X_test, y_train, y_test, is_mixed=False):
    # TODO: use Hyperopt for hyperparameter search
    # Split the dataset

    # X and y arrays must have matching numbers of rows
    # assert X.shape[0] == y.shape[0]

    scoring = CONFIG['train']['scoring']
    print("# Tuning hyper-parameters for %s (classifier: %s)" % (scoring, CONFIG['train']['classifier']))
    print()

    clf = build_grid(args, is_mixed=is_mixed)
    clf.fit(X_train, y_train)

    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()

    print("Best score: %s=%f" % (scoring, clf.best_score_))
    print()
    return clf


def feat_imp(args, y, X, num_features=25):
    import matplotlib.pyplot as plt
    clf = ExtraTreesClassifier(n_estimators=60)
    clf.fit(X, y)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for feat in xrange(num_features):
        print("%d. feature %d (%f)" % (feat + 1, indices[feat], importances[indices[feat]]))

    # Plot the feature importances of the clf
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_features), importances[indices][:num_features],
            yerr=std[indices][:num_features], align="center")
    plt.xticks(range(num_features), indices)
    plt.xlim([-1, num_features])
    plt.savefig(args.plot_features)


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--embedding', type=str, metavar='FILE', default=None,
                        help='Input word2vec (or doc2vec) model')
    parser.add_argument('--train', type=str, metavar='FILE', nargs='*', default=[],
                        help='(Labeled) training set')
    parser.add_argument('--plot_features', type=str, default=None,
                        help='file to save feature comparison to')
    parser.add_argument('--sentences', type=GzipFileType('r'), nargs='*', default=[],
                        help='File containing sentences in JSON format (implies doc2vec)')
    parser.add_argument('--vectors', metavar='FILE', type=str, default=None,
                        help='File containing sentence vectors in Pickle format')
    namespace = parser.parse_args(args)
    return namespace


def get_data(args):

    feature_set_names = CONFIG['train']['features']
    if set(feature_set_names).intersection(['word2vec', 'doc2vec']) and not args.embedding:
        raise RuntimeError("--embedding argument must be supplied")

    # get input data
    y_labels = np.array(list(pd_row_iter(args.train, field="sentiment")), dtype=float)
    sentences = [obj['review'] for obj in chain.from_iterable(read_json_lines(x) for x in args.sentences)]

    if not args.embedding or feature_set_names == ['bow']:
        # don't drop NaNs -- have a sparse matrix here
        return False, (get_bow_features(sentences), y_labels)

    # load embedding
    if CONFIG['pretrain']['algorithm'] == 'word2vec':
        from gensim.models import word2vec
        embedding = word2vec.Word2Vec.load(args.embedding)
    elif CONFIG['pretrain']['algorithm'] == 'glove':
        from glove import Glove
        embedding = Glove.load(args.embedding)
        # dynamicaly add GloveWrapper mixin
        embedding.__class__ = type('MyGlove', (Glove, GloveWrapper), {})

    # get feature vectors
    if 'doc2vec' in CONFIG['train']['features']:
        embedding_vectors = get_doc2vec_features(sentences, embedding)
    elif 'word2vec' in CONFIG['train']['features']:
        embedding_vectors = get_word2vec_features(sentences, embedding)
    else:
        raise RuntimeError("Invalid config setting train:features=%s" % CONFIG['train']['features'])

    if 'bow' in feature_set_names:
        return True, get_mixed_features(sentences, embedding_vectors, y_labels)
    else:
        # matrix is dense -- drop NaNs
        return False, drop_nans(embedding_vectors, y_labels)


def get_data_alt(args):
    with open_gz(args.vectors, "rb") as fh:
        train_X, train_y = pickle.load(fh)
        test_X, test_y = pickle.load(fh)
    vect = BagVectorizer().fit(train_X).fit(test_X)
    train_X = vect.transform(train_X)
    test_X = vect.transform(test_X)
    return train_X, test_X, np.asarray(train_y), np.asarray(test_y)


def run(args):

    if args.plot_features:
        assert not args.vectors
        data = get_data(args)
        is_mixed, (X, y) = data
        feat_imp(args, y, X)
    else:
        if args.vectors:
            data = False, get_data_alt(args)
            is_mixed, (X_train, X_test, y_train, y_test) = data
        else:
            data = get_data(args)
            is_mixed, (X, y) = data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0)
        train_model(args, X_train, X_test, y_train, y_test, is_mixed=is_mixed)


if __name__ == "__main__":
    run(parse_args())
