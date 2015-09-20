NLTK_DIR := nltk_data
NLTK_DIR_DONE := $(NLTK_DIR)/make.done

CFG := python -m flaubert.conf --key
EXT := $(shell $(CFG) data extension)
DATA_DIR := $(shell $(CFG) data directory)

TESTING_DATA := $(DATA_DIR)/$(shell $(CFG) data test_final)
TRAINING_LABELED := $(wildcard $(DATA_DIR)/$(shell $(CFG) data train_labeled).$(EXT))
TRAINING_UNLABELED :=  $(wildcard $(DATA_DIR)/$(shell $(CFG) data train_unlabeled).$(EXT))
TRAINING_ALL := $(TRAINING_LABELED) $(TRAINING_UNLABELED)
SENTENCE_LABELED := $(TRAINING_LABELED:.$(EXT)=.sents_labeled.gz)
SENTENCE_UNLABELED := $(TRAINING_UNLABELED:.$(EXT)=.sents_unlabeled.gz)
SENTENCE_ALL := $(SENTENCE_LABELED) $(SENTENCE_UNLABELED)

EMBEDDING := $(DATA_DIR)/300features_40minwords_10context
SENT_TOKENIZER := $(DATA_DIR)/sentence_tokenizer.pickle
CORP_MODEL := $(DATA_DIR)/glove-corpus.model
ROC_OUTPUT := $(DATA_DIR)/roc.png

export NLTK_DATA=$(NLTK_DIR)

.SECONDARY: $(SENT_TOKENIZER) $(SENTENCE_ALL) $(EMBEDDING)

clean_embedding:
	rm -rf $(EMBEDDING) $(EMBEDDING).syn0.npy

clean_sentences:
	rm -rf $(SENTENCE_ALL)

clean_data: clean_sentences clean_embedding

nltk: $(NLTK_DIR_DONE)
	@echo "done"

preprocess: $(SENTENCE_ALL) | env
	@echo "done"

pretrain: $(EMBEDDING)
	@echo "done"

train: $(TRAINING_LABELED) $(SENTENCE_LABELED) $(EMBEDDING)
	@echo "Training classifier"
	$(PYTHON) -m flaubert.train \
		--plot_roc $(ROC_OUTPUT) \
		--embedding $(EMBEDDING) \
		--train $(TRAINING_LABELED) \
		--sentences $(SENTENCE_LABELED)

train_vectors:
	$(PYTHON) -m flaubert.train --vectors data/imdb-old.pkl

%.$(EXT): %.$(EXT).zip
	unzip -p $< > $@

$(EMBEDDING): $(SENTENCE_ALL)
	@echo "Building embedding model at $(EMBEDDING)"
	python -m flaubert.pretrain --verbose \
		--input $^ \
		--corpus_model $(CORP_MODEL) \
		--output $@

$(NLTK_DIR_DONE):
	$(PYTHON) -m nltk.downloader -d $(NLTK_DIR) wordnet stopwords punkt maxent_treebank_pos_tagger
	touch $@

%.sents_labeled.gz: %.$(EXT) | $(NLTK_DIR_DONE) $(SENT_TOKENIZER)
	@echo "Building $@"
	$(PYTHON) -m flaubert.preprocess --input_labeled $*.$(EXT) --output $@ tokenize --sentences

%.sents_unlabeled.gz: %.$(EXT) | $(NLTK_DIR_DONE) $(SENT_TOKENIZER)
	@echo "Building $@"
	$(PYTHON) -m flaubert.preprocess --input_unlabeled $*.$(EXT) --output $@ tokenize --sentences

$(SENT_TOKENIZER): $(TRAINING_ALL)
	@echo "Building tokenizer at $@"
	$(PYTHON) -m flaubert.preprocess --input_labeled $(TRAINING_LABELED) --input_unlabeled $(TRAINING_UNLABELED) --output $@ train --verbose
