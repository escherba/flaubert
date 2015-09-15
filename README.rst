Flaubert
========

Sentiment analysis and word embedding.

This is trained on IMBD data (``prometheus:/media/data/escherba/flaubert/data``). It achieves about 88-90% accuracy with either bag of words (BoW) or embedding (word2vec) models.

For production use where we don't have labeled sets, labels could be generated from emoticons or emojis found in content (Happy = positive, sad = negative). This has been done previously by a researcher and is known to give good results given enough data.

Original data source: http://ai.stanford.edu/%7Eamaas/data/sentiment/

Running
-------

Data for this package are found on Prometheus box at ``/media/data/escherba/flaubert/data``.

Commands (make targets) for running various steps are found in ``analysis.mk``. The most general one is:

.. code-block:: bash

     make train
     
There is a lot of configuration in ``flaubert/conf/default.yaml``. Specifically, the following one is of interest:

.. code-block:: yaml

	train:
    	classifier: 'svm'
    	scoring: 'f1'
    	features: ["bow", "word2vec"]   # will use both BoW and word2vec features
    	nltk_stop_words: null    # either null or "english" or others.
 
"bow" means that a simple bag of words model will be used, "word2vec" means that word embeddings will be used.

In general, we first fit a sentence segmenter, then we train word vectors, then we plug them into an SVM.
