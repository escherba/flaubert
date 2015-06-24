#!/bin/bash

make env
make clean_data
rm -f data/300features_40minwords_10context
rm -f data/sentence_tokenizer.pickle
make data/sentence_tokenizer.pickle
make train
#make words
#make sentences
