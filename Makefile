# Makefile for Python Project

# Variables
PYTHON := python
PIP := pip
REQ := requirements.txt
SRC := ./src
PREPROCESS := $(SRC)/preprocessing.py
TEST := $(SRC)/testing.py
FASTTEXT := $(SRC)/fasttext_embedding.py
WORD2VEC := $(SRC)/word2vec_embedding.py
TFIDF := $(SRC)/tfidf_embedding.py
ML := $(SRC)/ml_models.py
MAIN := $(SRC)/main.py
GAN := $(SRC)/gan_augmentation.py

# Help
help:
	@echo "Available targets:"
	@echo "  all              Run full pipeline"
	@echo "  install          Install dependencies"
	@echo "  preprocess       Run preprocessing"
	@echo "  test             Run testing"
	@echo "  fasttext         Run FastText embeddings"
	@echo "  word2vec         Run Word2Vec embeddings"
	@echo "  tfidf            Run TF-IDF embeddings"
	@echo "  ml               Run all ML models (FastText)"
	@echo "  ml-[rf|xgb|knn|lr]     Run specific ML model (FastText)"
	@echo "  ml-word2vec      Run ML models (Word2Vec)"
	@echo "  ml-tfidf         Run ML models (TF-IDF)"
	@echo "  ml-w2v-[rf|xgb]  Run specific ML model (Word2Vec)"
	@echo "  ml-tfidf-[rf|xgb] Run specific ML model (TF-IDF)"
	@echo "  ml-evaluate      Evaluate without training"
	@echo "  ml-all-embeddings Run ML with all embeddings"
	@echo "  gan              Run GAN augmentation"
	@echo "  clean            Remove cache files"
	@echo "  clean-all        Remove all outputs + cache"
	@echo "  help             Show this message"

# Targets
all:        ; $(PYTHON) $(MAIN)
install:    ; $(PIP) install -r $(REQ)
preprocess: ; $(PYTHON) $(PREPROCESS)
test:       ; $(PYTHON) $(TEST)
fasttext:   ; $(PYTHON) $(FASTTEXT)
word2vec:   ; $(PYTHON) $(WORD2VEC)
tfidf:      ; $(PYTHON) $(TFIDF)

ml:         ; $(PYTHON) $(ML) --embedding-type fasttext
ml-rf:      ; $(PYTHON) $(ML) --model rf --embedding-type fasttext
ml-xgb:     ; $(PYTHON) $(ML) --model xgb --embedding-type fasttext
ml-knn:     ; $(PYTHON) $(ML) --model knn --embedding-type fasttext
ml-lr:      ; $(PYTHON) $(ML) --model lr --embedding-type fasttext

ml-word2vec:    ; $(PYTHON) $(ML) --embedding-type word2vec
ml-tfidf:       ; $(PYTHON) $(ML) --embedding-type tfidf
ml-w2v-rf:      ; $(PYTHON) $(ML) --model rf --embedding-type word2vec
ml-w2v-xgb:     ; $(PYTHON) $(ML) --model xgb --embedding-type word2vec
ml-tfidf-rf:    ; $(PYTHON) $(ML) --model rf --embedding-type tfidf
ml-tfidf-xgb:   ; $(PYTHON) $(ML) --model xgb --embedding-type tfidf

ml-all-embeddings:
	$(MAKE) ml
	$(MAKE) ml-word2vec
	$(MAKE) ml-tfidf

ml-evaluate: ; $(PYTHON) $(ML) --no-train
gan:         ; $(PYTHON) $(GAN)

clean:
	rm -rf __pycache__/ *.pyc
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

clean-all:
	rm -rf processed embeddings models results augmented
	$(MAKE) clean

.PHONY: all install preprocess test fasttext word2vec tfidf ml ml-rf ml-xgb ml-knn ml-lr \
        ml-word2vec ml-tfidf ml-w2v-rf ml-w2v-xgb ml-tfidf-rf ml-tfidf-xgb ml-all-embeddings \
        ml-evaluate gan clean clean-all help
