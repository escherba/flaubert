from __future__ import print_function

# from sklearn import datasets
import json
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from gensim.models import word2vec
# from scipy.stats import randint as sp_randint
# from sklearn.ensemble import RandomForestClassifier
from pymaptools.io import PathArgumentParser, GzipFileType
from nl2vec.preprocess import read_tsv

import numpy as np  # Make sure that numpy is imported


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


def classify(train, trainDataVecs):
    """
    :param train: Pandas data frame with training data
    :param trainDataVecs: a matrix of <samplex x features> dim.
    """
    X = trainDataVecs
    y = train["sentiment"]

    # Split the dataset in two parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # Set the parameters by cross-validation
    linearsvc_param_grid = [
        {'dual': [False], 'penalty':['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100, 1000]},
        {'dual': [True], 'penalty':['l2'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}
    ]

    # forest_param_grid = {
    #     "n_estimators": [10, 100],
    #     "max_depth": [10, 30, None],
    #     "max_features": [100, 300],
    #     "min_samples_split": [100, 300],
    #     "min_samples_leaf": [100, 300],
    #     "bootstrap": [False],
    #     "criterion": ["gini", "entropy"]
    # }

    score = 'f1'

    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(LinearSVC(), linearsvc_param_grid, cv=5,
                       scoring=score, n_jobs=6)
    # clf = GridSearchCV(RandomForestClassifier(), forest_param_grid, cv=5,
    #                 scoring=score, n_jobs=6)
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

    print("Best score: %s=%f" % (score, clf.best_score_))
    print()


def parse_args(args=None):
    parser = PathArgumentParser()
    parser.add_argument('--word2vec', type=str, metavar='FILE', required=True,
                        help='Input word2vec model')
    parser.add_argument('--train', type=str, metavar='FILE', required=True,
                        help='(Labeled) training set')
    parser.add_argument('--wordlist', type=GzipFileType('r'), required=True,
                        help='File containing words in JSON format')
    namespace = parser.parse_args(args)
    return namespace


def run(args):
    model = word2vec.Word2Vec.load(args.input)
    _, num_features = model.syn0.shape
    clean_train_reviews = []
    for line in args.wordlist:
        clean_train_reviews.append(json.loads(line))
    train = read_tsv(args.train)
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
    classify(train, trainDataVecs)


if __name__ == "__main__":
    run(parse_args())
