#!/bin/bash

make env
make nltk
#make clean_data
#rm -f data/sentence_tokenizer.pickle
#make data/sentence_tokenizer.pickle
#rm -f data/300features_40minwords_10context
#make preprocess
#make pretrain
make train_vectors
#make train
#make words
#make sentences
