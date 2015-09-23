from __future__ import print_function

import numpy as np
import logging
import cPickle as pickle
from copy import deepcopy
from itertools import chain, izip
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from pymaptools.io import PathArgumentParser, GzipFileType, read_json_lines, open_gz
from flaubert.pretrain import sentence_iter
from flaubert.utils import ItemSelector, BagVectorizer, drop_nans, reservoir_dict
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

    _, num_features = model.syn0.shape

    doc_sent_labels = list(sentence_iter(document_iter, cfg=CONFIG['train']))
    num_docs = len(doc_sent_labels)

    # Preallocate a 2D numpy array, for speed
    result = np.zeros((num_docs, num_features), dtype="float32")

    for idx, labels in enumerate(doc_sent_labels):
        nvecs = 0
        vector = np.zeros((num_features,), dtype="float32")
        for sentence, this_labels in labels:
            this_vector = makeFeatureVec(sentence, model, num_features)
            vector = np.add(vector, this_vector)
            nvecs += 1
        vector = np.divide(vector, float(nvecs))
        result[idx] = vector

    return result


def get_bow_features(documents):
    data = [list(chain(*doc)) for doc in documents]
    vectorizer = BagVectorizer(onehot=False, stop_words=STOP_WORDS)
    train_data_features = vectorizer.fit_transform(data)
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


TRANSFORMER_PARAMS = {
    None: [None, {}],
    'TfidfTransformer': [
        TfidfTransformer,
        {
            # 'tfidf__smooth_idf': [True, False],   # causes error in some cases
            'trans__norm': ['l1', 'l2', None],
            'trans__use_idf': [True, False],
            'trans__sublinear_tf': [True, False],
        }
    ]
}

CLF_PARAMS = {
    'MultinomialNB': [
        MultinomialNB,
        {
            'clf__alpha': [0.464, 1.0, 2.15, 4.64, 10.0, 21.5, 46.4]
        },
    ],
    'LogisticRegression': [
        LogisticRegression,
        [
            {'clf__dual': [False], 'clf__penalty':['l1', 'l2'], 'clf__C': [0.01, 0.1, 1, 10]},
            {'clf__dual': [True],  'clf__penalty':['l2'],       'clf__C': [0.01, 0.1, 1, 10]}
        ]
    ],
    'LinearSVC': [
        LinearSVC,
        [
            {'clf__dual': [False], 'clf__penalty':['l1', 'l2'], 'clf__C': [0.1, 1, 10, 100, 1000]},
            {'clf__dual': [True],  'clf__penalty':['l2'],       'clf__C': [0.1, 1, 10, 100, 1000]}
        ]
    ],
    'RandomForest': [
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
    'AdaBoost': [
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


def update_params(orig, copied):
    if isinstance(orig, list):
        new_params = []
        for orig_elem in orig:
            param_dict = deepcopy(orig_elem)
            param_dict.update(copied)
            new_params.append(param_dict)
        return new_params
    elif isinstance(orig, dict):
        new_params = deepcopy(orig)
        new_params.update(copied)
        return new_params
    else:
        raise TypeError("first parameter must be either list or dict")


def build_grid(args, is_mixed):
    clf, pipeline_params = CLF_PARAMS[CONFIG['train']['classifier']]
    feature_set_names = CONFIG['train']['features']
    if is_mixed:
        transformer_list = []
        transformer_weights = {}
        if set(feature_set_names).intersection(['embedding']):
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
    elif set(feature_set_names).intersection(['embedding']):
        pipeline = Pipeline([
            ('clf', clf()),
        ])
    elif set(feature_set_names).intersection(['bow']):
        steps = []
        trans, trans_params = TRANSFORMER_PARAMS[CONFIG['train']['transformer']]
        if trans:
            steps.append(('trans', trans()))
            pipeline_params = update_params(pipeline_params, trans_params)
        steps.append(('clf', clf()))
        pipeline = Pipeline(steps)

    grid_args = [pipeline, pipeline_params]
    grid_kwargs = dict(cv=5, scoring=CONFIG['train']['scoring'], n_jobs=-1, verbose=10)
    result = GridSearchCV(*grid_args, **grid_kwargs)
    return result


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

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()
    print("Best score from the development set:")
    print("%s=%f" % (scoring, clf.best_score_))
    print()

    print("Classification report on the evaluation set:")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    try:
        pred_probas = clf.predict_log_proba(X_test)
        pred_probas = pred_probas[:, 1]
    except AttributeError:
        pred_probas = clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC = %.3f" % roc_auc)

    if args.plot_roc:
        plot_roc(args.plot_roc, fpr, tpr, roc_auc)

    return clf


def plot_roc(output_file, fpr, tpr, roc_auc):
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr, label='AUC = %.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right')
    plt.savefig(output_file)


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
    parser.add_argument('--plot_features', type=str, default=None,
                        help='file to save feature comparison to')
    parser.add_argument('--plot_roc', type=str, default=None,
                        help='file to save feature comparison to')
    parser.add_argument('--sentences', type=GzipFileType('r'), nargs='*', default=[],
                        help='File containing sentences in JSON format (implies doc2vec)')
    parser.add_argument('--vectors', metavar='FILE', type=str, default=None,
                        help='File containing sentence vectors in Pickle format')
    namespace = parser.parse_args(args)
    return namespace


def sample_by_y(args):
    sample = chain.from_iterable(read_json_lines(x) for x in args.sentences)
    cfg = CONFIG['train']
    label_counts = cfg.get('sample_labeled')
    if label_counts:
        sample = reservoir_dict(sample, "Y", label_counts,
                                random_state=cfg['random_state'])
    sentences, yvals = zip(*[(obj['X'], obj['Y']) for obj in sample])
    y_labels = np.array(yvals, dtype=float)
    return sentences, y_labels


def get_data(args):

    feature_set_names = CONFIG['train']['features']
    if set(feature_set_names).intersection(['embedding']) and not args.embedding:
        raise RuntimeError("--embedding argument must be supplied")

    # get input data
    documents, y_labels = sample_by_y(args)

    if not args.embedding or feature_set_names == ['bow']:
        # don't drop NaNs -- have a sparse matrix here
        X = get_bow_features(documents)
        return False, (X, y_labels)

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
    if 'embedding' in CONFIG['train']['features']:
        embedding_vectors = get_word2vec_features(documents, embedding)
    else:
        raise RuntimeError("Invalid config setting train:features=%s" % CONFIG['train']['features'])

    if 'bow' in feature_set_names:
        X, y_labels = get_mixed_features(documents, embedding_vectors, y_labels)
        return True, (X, y_labels)
    else:
        # matrix is dense -- drop NaNs
        X, y_labels = drop_nans(embedding_vectors, y_labels)
        return False, (X, y_labels)


def get_data_alt(args):
    with open_gz(args.vectors, "rb") as fh:
        train_X, train_y = pickle.load(fh)
        test_X, test_y = pickle.load(fh)
    vect = BagVectorizer().fit(train_X).fit(test_X)
    train_X = vect.transform(train_X)
    test_X = vect.transform(test_X)
    return train_X, test_X, np.asarray(train_y), np.asarray(test_y)


def run(args):

    cfg = CONFIG['train']

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
                X, y, test_size=cfg['test_size'],
                random_state=cfg['random_state'])
        train_model(args, X_train, X_test, y_train, y_test, is_mixed=is_mixed)


if __name__ == "__main__":
    run(parse_args())
