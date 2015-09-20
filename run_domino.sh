#!/bin/bash

#make clean_data
#make nuke
make env
. env/bin/activate; pip install git+https://github.com/maciejkula/glove-python#egg=glove_python-0.0.1

make nltk

#rm -f data/aclImdb/sentence_tokenizer.pickle
#make ./data/aclImdb/sentence_tokenizer.pickle
#rm -f data/aclImdb/300features_40minwords_10context

make -j2 preprocess
make -j2 pretrain

#make train_vectors
make train
#make words
#make sentences
