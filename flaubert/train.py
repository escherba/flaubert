from __future__ import print_function

import numpy as np
import logging
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
    index2word_set = set(model.index2word)
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            vector = np.add(vector, model[word])
    # Divide the result by the number of words to get the average
    vector = np.divide(vector, float(nwords))
    return vector


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    counter = 0
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    # Loop through the reviews
    for review in reviews:
        # Print a status message every 1000th review
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        # Increment the counter
        counter = counter + 1
    return reviewFeatureVecs


SCORING = 'f1'

PARAM_GRIDS = {
    'LogisticRegression': [
        {'dual': [False], 'penalty':['l1', 'l2'], 'C': [0.01, 0.033, 0.1, 0.33, 1.0]},
        # {'dual': [True], 'penalty':['l2'], 'C': [0.01, 0.033, 0.1, 0.33, 1.0]}
    ],
    'LinearSVC': [
        {'dual': [False], 'penalty':['l1', 'l2'], 'C': [1.0, 3.3, 10.0, 30.0]},
        {'dual': [True], 'penalty':['l2'], 'C': [0.1, 1, 10, 100]}
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

    for f in xrange(num_features):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

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
    parser.add_argument('--word2vec', type=str, metavar='FILE', required=True,
                        help='Input word2vec model')
    parser.add_argument('--train', type=str, metavar='FILE', required=True,
                        help='(Labeled) training set')
    parser.add_argument('--plot_features', type=str, default=None,
                        help='file to save feature comparison to')
    parser.add_argument('--wordlist', type=GzipFileType('r'), required=True,
                        help='File containing words in JSON format')
    namespace = parser.parse_args(args)
    return namespace


def run(args, model=None):
    if model is None:
        model = word2vec.Word2Vec.load(args.word2vec)
    _, num_features = model.syn0.shape
    clean_train_reviews = list(read_json_lines(args.wordlist))
    training_set = read_tsv(args.train)
    feature_vectors = getAvgFeatureVecs(clean_train_reviews, model, num_features)
    if args.plot_features:
        feat_imp(args, training_set["sentiment"], feature_vectors)
    else:
        train_model(args, training_set["sentiment"], feature_vectors)


if __name__ == "__main__":
    run(parse_args())
