ifeq ($(DOMINO_RUN),1)
PYENV =
else
PYENV = . env/bin/activate;
endif
PYTHON = $(PYENV) python
PIP = $(PYENV) pip

CONFIG = ./nl2vec/conf/default.yaml
DATA_DIR = data
ALL_DATA := $(shell find $(DATA_DIR) -type f -name '*.zip')
TEST = $(DATA_DIR)/testData.tsv.zip
LABELED_TRAIN = $(DATA_DIR)/labeledTrainData.tsv.zip
UNLABELED_TRAIN =  $(DATA_DIR)/unlabeledTrainData.tsv.zip
TRAIN = $(LABELED_TRAIN) $(UNLABELED_TRAIN)

sentences: $(ALL_DATA:.tsv.zip=.sents.gz)
	@echo "done"

words: $(ALL_DATA:.tsv.zip=.words.gz)
	@echo "done"

%.sents.gz: %.tsv.zip | $(CONFIG)
	$(PYTHON) -m nl2vec.preprocess --sentences --input $^ --output $@

%.words.gz: %.tsv.zip | $(CONFIG)
	$(PYTHON) -m nl2vec.preprocess --input $^ --output $@
