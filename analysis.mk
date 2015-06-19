CONFIG = ./nl2vec/conf/default.yaml
NLTK_DIR = nltk_data
NLTK_DIR_DONE = $(NLTK_DIR)/make.done
DATA_DIR = data
ALL_DATA := $(shell find $(DATA_DIR) -type f -name '*.zip')
TEST = $(DATA_DIR)/testData
LABELED_TRAIN = $(DATA_DIR)/labeledTrainData
UNLABELED_TRAIN =  $(DATA_DIR)/unlabeledTrainData
TRAIN = $(LABELED_TRAIN) $(UNLABELED_TRAIN)
WORD2VEC = $(DATA_DIR)/300features_40minwords_10context.2

export NLTK_DATA=$(NLTK_DIR)

SENTS := $(ALL_DATA:.tsv.zip=.sents.gz)
WORDS := $(ALL_DATA:.tsv.zip=.words.gz)

clean_data:
	rm -rf $(SENTS) $(WORDS)

sentences: $(SENTS) | env
	@echo "done"

words: $(WORDS) | env
	@echo "done"

nltk: $(NLTK_DIR_DONE)
	@echo "done"

pretrain: $(WORD2VEC)
	@echo "done"

train: $(LABELED_TRAIN).tsv.zip $(LABELED_TRAIN).words.gz $(WORD2VEC)
	unzip -p $(LABELED_TRAIN).tsv.zip > $(LABELED_TRAIN).tsv
	$(PYTHON) -m nl2vec.classify \
		--classifier svm --word2vec $(WORD2VEC) \
		--train $(LABELED_TRAIN).tsv --wordlist $(LABELED_TRAIN).words.gz
	rm -f $(LABELED_TRAIN).tsv

$(WORD2VEC): $(LABELED_TRAIN).sents.gz $(UNLABELED_TRAIN).sents.gz
	python -m nl2vec.pretrain \
		--sentences $^ \
		--output $@

$(NLTK_DIR_DONE):
	$(PYTHON) -m nltk.downloader -d $(NLTK_DIR) wordnet stopwords punkt maxent_treebank_pos_tagger
	touch $@

%.sents.gz: %.tsv.zip | $(CONFIG) $(NLTK_DIR_DONE)
	unzip -p $< > $*.tsv
	$(PYTHON) -m nl2vec.preprocess --sentences --input $*.tsv --output $@

%.words.gz: %.tsv.zip | $(CONFIG) $(NLTK_DIR_DONE)
	unzip -p $< > $*.tsv
	$(PYTHON) -m nl2vec.preprocess --input $*.tsv --output $@
