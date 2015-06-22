LinearSVC
=========


The model is trained on the full development set.
The scores are computed on the full evaluation set.

             precision    recall  f1-score   support

          0       0.89      0.88      0.88      3789
          1       0.87      0.88      0.88      3711

avg / total       0.88      0.88      0.88      7500


Best parameters set found on development set:

{'penalty': 'l2', 'C': 10, 'dual': False}

Best score: f1=0.873566


RandomForestClassifier
======================

The model is trained on the full development set.
The scores are computed on the full evaluation set.

             precision    recall  f1-score   support

          0       0.84      0.81      0.83      3789
          1       0.81      0.84      0.83      3711

avg / total       0.83      0.83      0.83      7500


Best parameters set found on development set:

{'bootstrap': False, 'min_samples_leaf': 30, 'min_samples_split': 30, 'criterion': 'entropy', 'max_features': 30, 'max_depth': None}

Best score: f1=0.822510

After adding emoji and rating support
=====================================

The model is trained on the full development set.
The scores are computed on the full evaluation set.

             precision    recall  f1-score   support

          0       0.89      0.88      0.88      2548
          1       0.87      0.89      0.88      2452

avg / total       0.88      0.88      0.88      5000


Best parameters set found on development set:

{'penalty': 'l2', 'C': 10, 'dual': False}

Best score: f1=0.872732

BoW
===


SVM
---

             precision    recall  f1-score   support

          0       0.89      0.87      0.88      2548
          1       0.87      0.88      0.88      2452

avg / total       0.88      0.88      0.88      5000


Best parameters set found on development set:

{'penalty': 'l1', 'C': 0.1, 'dual': False}

Best score: f1=0.880630


Logistic Regression
-------------------

Detailed classification report:

The model is trained on the full development set.
The scores are computed on the full evaluation set.

             precision    recall  f1-score   support

          0       0.90      0.88      0.89      2548
          1       0.88      0.89      0.89      2452

avg / total       0.89      0.89      0.89      5000


Best parameters set found on development set:

{'penalty': 'l2', 'C': 0.1, 'dual': False}

Best score: f1=0.884835
