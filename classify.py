from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier


n_samples = len(trainDataVecs)
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

forest_param_grid = {
    "n_estimators": [10, 100],
    "max_depth": [10, 30, None],
    "max_features": [100, 300],
    "min_samples_split": [100, 300],
    "min_samples_leaf": [100, 300],
    "bootstrap": [False],
    "criterion": ["gini", "entropy"]
}

scores = ['f1']

print("# Tuning hyper-parameters for %s" % score)
print()

#clf = GridSearchCV(LinearSVC(), linearsvc_param_grid, cv=5,
#                   scoring=score, n_jobs=6)
clf = GridSearchCV(RandomForestClassifier(), forest_param_grid, cv=5,
                   scoring=score, n_jobs=6)
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
