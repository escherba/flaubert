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
