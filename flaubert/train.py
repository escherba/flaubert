from __future__ import print_function

import numpy as np
import logging
from itertools import chain
from gensim.models import word2vec
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from pymaptools.io import PathArgumentParser, GzipFileType, read_json_lines
from flaubert.preprocess import read_tsv
from flaubert.pretrain import sentence_iter


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    vector = np.zeros((num_features,), dtype="float32")
    nwords = 0
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        try:
            word_vector = model[word]
        except KeyError:
            continue
        vector = np.add(vector, word_vector)
        nwords = nwords + 1
    # Divide the result by the number of words to get the average
    vector = np.divide(vector, float(nwords))
    return vector


def getAvgFeatureVecs(wordlist_file, model):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    _, num_features = model.syn0.shape

    reviews = list(read_json_lines(wordlist_file))
    num_reviews = len(reviews)

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((num_reviews, num_features), dtype="float32")
    for idx, review in enumerate(reviews):
        if (idx + 1) % 1000 == 0:
            print("Review %d of %d" % ((idx + 1), num_reviews))
        reviewFeatureVecs[idx] = makeFeatureVec(review, model, num_features)
    return reviewFeatureVecs


def getDoc2VecVectors(sentence_file, model):

    _, num_features = model.syn0.shape

    review_sent_labels = list(sentence_iter([sentence_file]))
    num_reviews = len(review_sent_labels)

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((num_reviews, num_features), dtype="float32")

    for idx, labels in enumerate(review_sent_labels):
        if (idx + 1) % 1000 == 0:
            print("Review %d of %d" % ((idx + 1), num_reviews))
        nvecs = 0
        vector = np.zeros((num_features,), dtype="float32")
        for sentence, this_labels in labels:
            this_vector = makeFeatureVec(chain(sentence, this_labels), model, num_features)
            vector = np.add(vector, this_vector)
            nvecs += 1
        vector = np.divide(vector, float(nvecs))
        reviewFeatureVecs[idx] = vector

    return reviewFeatureVecs


SCORING = 'f1'

PARAM_GRIDS = {
    'LogisticRegression': [
        {'dual': [False], 'penalty':['l1', 'l2'], 'C': [0.01, 0.033, 0.1, 0.33, 1.0]},
        # {'dual': [True], 'penalty':['l2'], 'C': [0.01, 0.033, 0.1, 0.33, 1.0]}
    ],
    'LinearSVC': [
        {'dual': [False], 'penalty':['l1', 'l2'], 'C': [1, 3.33, 10, 33, 100, 333]},
        # {'dual': [True], 'penalty':['l2'], 'C': [0.1, 1, 10, 100]}
    ],
    'RandomForestClassifier': {
        "n_estimators": [90],
        "max_depth": [32, 64],
        "max_features": [50, 75, 100],
        "min_samples_split": [2],
        "min_samples_leaf": [2, 3],
        "bootstrap": [False],
        "criterion": ["gini"]
    },
    'AdaBoost': {
        'n_estimators': [60],
        'learning_rate': [0.8],
        'algorithm': ['SAMME.R']
    }
}

GRIDSEARHCV_KWARGS = dict(cv=5, scoring=SCORING, n_jobs=-1, verbose=10)
DECISION_TREE_PARAMS = dict(
    criterion="gini", max_depth=2, min_samples_split=2, min_samples_leaf=2
)

CLASSIFIER_GRIDS = {
    'lr': [[LogisticRegression(), PARAM_GRIDS['LogisticRegression']], GRIDSEARHCV_KWARGS],
    'svm': [[LinearSVC(), PARAM_GRIDS['LinearSVC']], GRIDSEARHCV_KWARGS],
    'random_forest': [[RandomForestClassifier(), PARAM_GRIDS['RandomForestClassifier']], GRIDSEARHCV_KWARGS],
    'adaboost': [[AdaBoostClassifier(DecisionTreeClassifier(**DECISION_TREE_PARAMS)), PARAM_GRIDS['AdaBoost']], GRIDSEARHCV_KWARGS]
}


def train_model(args, y, X):
    # TODO: use Hyperopt for hyperparameter search
    # Split the dataset

    # X and y arrays must have matching numbers of rows
    assert X.shape[0] == y.shape[0]

    # drop rows that contain any NaNs (missing values)
    X_nans = np.isnan(X).any(axis=1)
    y_nans = np.asarray(np.isnan(y))
    nans = X_nans | y_nans
    y = y[~nans]
    X = X[~nans]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    print("# Tuning hyper-parameters for %s (classifier: %s)" % (SCORING, args.classifier))
    print()

    args, kwargs = CLASSIFIER_GRIDS[args.classifier]
    clf = GridSearchCV(*args, **kwargs)
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

    print("Best score: %s=%f" % (SCORING, clf.best_score_))
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
    parser.add_argument('--classifier', type=str, default='svm',
                        choices=CLASSIFIER_GRIDS.keys(),
                        help='Which classifier to use')
    parser.add_argument('--model', type=str, metavar='FILE', default=None,
                        help='Input word2vec (or doc2vec) model')
    parser.add_argument('--train', type=str, metavar='FILE', required=True,
                        help='(Labeled) training set')
    parser.add_argument('--plot_features', type=str, default=None,
                        help='file to save feature comparison to')
    parser.add_argument('--sentencelist', type=GzipFileType('r'), default=None,
                        help='File containing sentences in JSON format (implies doc2vec)')
    parser.add_argument('--wordlist', type=GzipFileType('r'), default=None,
                        help='File containing words in JSON format (implies word2vec)')
    namespace = parser.parse_args(args)
    return namespace


def run(args, model=None):

    # load model
    if model is None:
        model = word2vec.Word2Vec.load(args.model)

    # get feature vectors
    if args.sentencelist:
        feature_vectors = getDoc2VecVectors(args.sentencelist, model)
    elif args.wordlist:
        feature_vectors = getAvgFeatureVecs(args.wordlist, model)
    else:
        raise RuntimeError("Either word list or sentence list must be specified")

    # get Y labels
    training_set = read_tsv(args.train)
    y_labels = training_set["sentiment"]

    # train a classifier
    if args.plot_features:
        feat_imp(args, y_labels, feature_vectors)
    else:
        train_model(args, y_labels, feature_vectors)


if __name__ == "__main__":
    run(parse_args())
