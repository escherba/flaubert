NLTK_DIR = nltk_data
NLTK_DIR_DONE = $(NLTK_DIR)/make.done
DATA_DIR = data
ALL_DATA := $(shell find $(DATA_DIR) -type f -name '*.zip')
TEST = $(DATA_DIR)/testData
LABELED_TRAIN = $(DATA_DIR)/labeledTrainData
UNLABELED_TRAIN =  $(DATA_DIR)/unlabeledTrainData
TRAIN = $(LABELED_TRAIN) $(UNLABELED_TRAIN)
EMBEDDING = $(DATA_DIR)/300features_40minwords_10context
SENT_TOKENIZER = $(DATA_DIR)/sentence_tokenizer.pickle

export NLTK_DATA=$(NLTK_DIR)

TSVS  := $(ALL_DATA:.tsv.zip=.tsv)
SENTS := $(ALL_DATA:.tsv.zip=.sents.gz)

clean_data:
	rm -rf $(SENTS) $(WORDS) $(EMBEDDING)

nltk: $(NLTK_DIR_DONE)
	@echo "done"

preprocess: $(SENTS) | env
	@echo "done"

pretrain: $(EMBEDDING)
	@echo "done"

train: $(LABELED_TRAIN).tsv $(LABELED_TRAIN).sents.gz $(EMBEDDING)
	$(PYTHON) -m flaubert.train \
		--embedding $(EMBEDDING) \
		--train $(LABELED_TRAIN).tsv \
		--sentences $(LABELED_TRAIN).sents.gz

.SECONDARY: $(TSVS) $(SENT_TOKENIZER) $(WORDS) $(SENTS) $(EMBEDDING)
%.tsv: %.tsv.zip
	unzip -p $< > $@

$(EMBEDDING): $(LABELED_TRAIN).sents.gz $(UNLABELED_TRAIN).sents.gz
	@echo "Building embedding model at $(EMBEDDING)"
	python -m flaubert.pretrain \
		--sentences $^ \
		--output $@

$(NLTK_DIR_DONE):
	$(PYTHON) -m nltk.downloader -d $(NLTK_DIR) wordnet stopwords punkt maxent_treebank_pos_tagger
	touch $@

%.sents.gz: %.tsv | $(NLTK_DIR_DONE) $(SENT_TOKENIZER)
	$(PYTHON) -m flaubert.preprocess --input $*.tsv --output $@ tokenize --sentences

$(SENT_TOKENIZER): $(LABELED_TRAIN).tsv $(UNLABELED_TRAIN).tsv
	$(PYTHON) -m flaubert.preprocess --input $^ --output $@ train --verbose
