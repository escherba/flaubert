ifeq ($(DOMINO_RUN),1)
PYENV =
else
PYENV = . env/bin/activate;
endif
PYTHON = $(PYENV) python
PIP = $(PYENV) pip

CONFIG = ./nl2vec/conf/default.yaml
NLTK_DIR = nltk_data
NLTK_DIR_DONE = $(NLTK_DIR)/make.done
DATA_DIR = data
ALL_DATA := $(shell find $(DATA_DIR) -type f -name '*.zip')
TEST = $(DATA_DIR)/testData.tsv.zip
LABELED_TRAIN = $(DATA_DIR)/labeledTrainData.tsv.zip
UNLABELED_TRAIN =  $(DATA_DIR)/unlabeledTrainData.tsv.zip
TRAIN = $(LABELED_TRAIN) $(UNLABELED_TRAIN)

export NLTK_DATA=$(NLTK_DIR)

sentences: $(ALL_DATA:.tsv.zip=.sents.gz)
	@echo "done"

words: $(ALL_DATA:.tsv.zip=.words.gz)
	@echo "done"

nltk: $(NLTK_DIR_DONE)
	@echo "done"

$(NLTK_DIR_DONE):
	$(PYTHON) -m nltk.downloader -d $(NLTK_DIR) wordnet stopwords punkt maxent_treebank_pos_tagger
	touch $@

%.sents.gz: %.tsv.zip | $(CONFIG) $(NLTK_DIR_DONE)
	$(PYTHON) -m nl2vec.preprocess --sentences --input $^ --output $@

%.words.gz: %.tsv.zip | $(CONFIG) $(NLTK_DIR_DONE)
	$(PYTHON) -m nl2vec.preprocess --input $^ --output $@
