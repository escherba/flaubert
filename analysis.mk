CONFIG = ./nl2vec/conf/default.yaml
NLTK_DIR = nltk_data
NLTK_DIR_DONE = $(NLTK_DIR)/make.done
DATA_DIR = data
ALL_DATA := $(shell find $(DATA_DIR) -type f -name '*.zip')
TEST = $(DATA_DIR)/testData.tsv
LABELED_TRAIN = $(DATA_DIR)/labeledTrainData.tsv
UNLABELED_TRAIN =  $(DATA_DIR)/unlabeledTrainData.tsv
TRAIN = $(LABELED_TRAIN) $(UNLABELED_TRAIN)

export NLTK_DATA=$(NLTK_DIR)

SENTS := $(ALL_DATA:.tsv.zip=.sents.gz)
WORDS := $(ALL_DATA:.tsv.zip=.words.gz)

clean_data:
	rm -rf $(SENTS) $(WORDS)

sentences: $(SENTS)
	@echo "done"

words: $(WORDS)
	@echo "done"

nltk: $(NLTK_DIR_DONE)
	@echo "done"

$(NLTK_DIR_DONE):
	$(PYTHON) -m nltk.downloader -d $(NLTK_DIR) wordnet stopwords punkt maxent_treebank_pos_tagger
	touch $@

%.tsv: %.tsv.zip
	unzip -p $^ > $@

%.sents.gz: %.tsv | $(CONFIG) $(NLTK_DIR_DONE)
	$(PYTHON) -m nl2vec.preprocess --sentences --input $^ --output $@

%.words.gz: %.tsv | $(CONFIG) $(NLTK_DIR_DONE)
	$(PYTHON) -m nl2vec.preprocess --input $^ --output $@
