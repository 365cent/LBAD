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
XGBOOST_ML := $(SRC)/xgboost_ml.py
MAIN := $(SRC)/main.py
GAN := $(SRC)/gan_augmentation.py
EVAL := $(SRC)/gan_evaluation.py

# Log types for specific processing
# From src/preprocessing.py --log-type choices
PREPROCESS_LOG_TYPES := web error auth dns firewall vpn audit kernel systemd
# From src/fasttext_embedding.py LOG_TYPE_ATTACKS.keys()
FASTTEXT_LOG_TYPES := dns network web error monitoring auth audit

# Contexts for xgboost_ml.py (specific logical types + all_combined)
XGBOOST_ML_CONTEXTS := all_combined $(FASTTEXT_LOG_TYPES)

# Help
help:
	@echo "Available targets:"
	@echo "  all              Run full pipeline"
	@echo "  install          Install dependencies"
	@echo "  preprocess       Run preprocessing for ALL raw log types (creates 'processed/<type>/...')"
	@echo "  preprocess-<type>  Run preprocessing for a specific raw log type (e.g., make preprocess-web)"
	@echo "                     Creates 'processed/<type>/...' outputs."
	@echo "                     Available types for preprocess: $(PREPROCESS_LOG_TYPES)"
	@echo "  test             Run testing"
	@echo "  fasttext         Run FastText embeddings for ALL defined logical log types"
	@echo "                     (processes data from 'processed/<type>/...', creates 'embeddings/<type>/...')"
	@echo "  fasttext-<type>    Run FastText for a specific logical log type (e.g., make fasttext-network)"
	@echo "                     Creates 'embeddings/<type>/...' outputs."
	@echo "                     Available types for fasttext: $(FASTTEXT_LOG_TYPES)"
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
	@echo "  xgboost-ml         Run XGBoost Multi-Label (OvR) for all_combined context"
	@echo "  xgboost-ml-<ctx>   Run XGBoost Multi-Label (OvR) for a specific context (e.g., make xgboost-ml-web)"
	@echo "                     Available contexts for xgboost-ml: $(XGBOOST_ML_CONTEXTS)"
	@echo "  gan              Run VAE-GAN augmentation"
	@echo "  eval             Evaluate VAE-GAN augmentation"
	@echo "  gan-pipeline     Run augmentation and evaluation"
	@echo "  clean            Remove cache files"
	@echo "  clean-all        Remove all outputs + cache"
	@echo "  clean-aug        Remove augmented data only"
	@echo "  help             Show this message"

# Targets
all:        ; $(PYTHON) $(MAIN)
install:    ; $(PIP) install -r $(REQ)

# Preprocessing: 'make preprocess' processes all types into subdirectories.
preprocess: ; $(PYTHON) $(PREPROCESS)

# Preprocessing by specific log type (e.g., make preprocess-web)
# Outputs to 'processed/<type>/'
preprocess-%:
	@echo "Preprocessing specific log type: $* (from 'logs/' to 'processed/$*/')"
	$(PYTHON) $(PREPROCESS) --log-type $*

test:       ; $(PYTHON) $(TEST)

# FastText: 'make fasttext' processes all logical types from 'processed/<type>/' to 'embeddings/<type>/'.
fasttext:   ; $(PYTHON) $(FASTTEXT)

# FastText embedding by specific logical log type (e.g., make fasttext-network)
# Outputs to 'embeddings/<type>/'
fasttext-%:
	@echo "Running FastText embedding for specific logical log type: $* (from 'processed/' to 'embeddings/$*/')"
	$(PYTHON) $(FASTTEXT) --log-type $*

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

# New targets for XGBoost Multi-Label (OvR) baseline
xgboost-ml:
	@echo "Running XGBoost Multi-Label (OvR) for context: all_combined"
	$(PYTHON) $(XGBOOST_ML) --context all_combined

xgboost-ml-%:
	@echo "Running XGBoost Multi-Label (OvR) for context: $*"
	$(PYTHON) $(XGBOOST_ML) --context $*

ml-evaluate: ; $(PYTHON) $(ML) --no-train
gan:         ; $(PYTHON) $(GAN)
eval:        ; $(PYTHON) $(EVAL)

# New targets for VAE-GAN pipeline
gan-pipeline:
	$(MAKE) gan
	$(MAKE) eval

gan-small:   ; $(PYTHON) $(GAN) --epochs 20 --threshold 0.05
eval-fast:   ; $(PYTHON) $(EVAL)

clean:
	rm -rf __pycache__/ *.pyc
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

clean-aug:
	rm -rf augmented
	mkdir -p augmented

clean-all:
	rm -rf processed embeddings models results augmented
	mkdir -p processed embeddings models results augmented
	$(MAKE) clean

.PHONY: all install preprocess test fasttext word2vec tfidf ml ml-rf ml-xgb ml-knn ml-lr \
        ml-word2vec ml-tfidf ml-w2v-rf ml-w2v-xgb ml-tfidf-rf ml-tfidf-xgb ml-all-embeddings \
        ml-evaluate xgboost-ml gan eval gan-pipeline gan-small eval-fast clean clean-aug clean-all help
# Add xgboost-ml-% to .PHONY if it doesn't create a file named xgboost-ml-<ctx>
# For pattern rules, if they always execute commands and don't create a target file,
# they should be .PHONY. However, make often treats pattern rules as .PHONY by default
# if no corresponding file is ever created by them.
# Explicitly adding it for clarity or if issues arise:
.PHONY: $(addprefix xgboost-ml-,$(XGBOOST_ML_CONTEXTS))
