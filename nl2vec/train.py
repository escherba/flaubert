from __future__ import print_function

import json
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from gensim.models import word2vec
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from pymaptools.io import PathArgumentParser, GzipFileType
from nl2vec.preprocess import read_tsv


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


PARAM_GRIDS = {
    'LinearSVC': [
        {'dual': [False], 'penalty':['l1', 'l2'], 'C': [0.1, 1, 10, 100]},
        # {'dual': [True], 'penalty':['l2'], 'C': [0.1, 1, 10, 100]}
    ],
    'RandomForestClassifier': {
        "n_estimators": [30, 40],
        "max_depth": [10, 30],
        "max_features": [50, 100],
        "min_samples_split": [100],
        "min_samples_leaf": [100],
        "bootstrap": [False],
        "criterion": ["gini", "entropy"]
    }
}

SCORING = 'f1'

CLASSIFIER_GRIDS = {
    'svm': [[LinearSVC(), PARAM_GRIDS['LinearSVC']],
            dict(cv=5, scoring=SCORING, n_jobs=6)],
    'random_forest': [[RandomForestClassifier(), PARAM_GRIDS['RandomForestClassifier']],
                      dict(cv=5, scoring=SCORING, n_jobs=6)],
}


def train_model(classifier, y, X):
    # TODO: use Hyperopt for hyperparameter search
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    print("# Tuning hyper-parameters for %s (classifier: %s)" % (SCORING, classifier))
    print()

    args, kwargs = CLASSIFIER_GRIDS[classifier]
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


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--classifier', type=str, default='svm',
                        choices=CLASSIFIER_GRIDS.keys(),
                        help='Which classifier to use')
    parser.add_argument('--word2vec', type=str, metavar='FILE', required=True,
                        help='Input word2vec model')
    parser.add_argument('--train', type=str, metavar='FILE', required=True,
                        help='(Labeled) training set')
    parser.add_argument('--wordlist', type=GzipFileType('r'), required=True,
                        help='File containing words in JSON format')
    namespace = parser.parse_args(args)
    return namespace


def run(args):
    model = word2vec.Word2Vec.load(args.word2vec)
    _, num_features = model.syn0.shape
    clean_train_reviews = []
    for line in args.wordlist:
        clean_train_reviews.append(json.loads(line))
    training_set = read_tsv(args.train)
    feature_vectors = getAvgFeatureVecs(clean_train_reviews, model, num_features)
    train_model(args.classifier, training_set["sentiment"], feature_vectors)


if __name__ == "__main__":
    run(parse_args())
