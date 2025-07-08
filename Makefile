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
LOGBERT_EMBEDDINGS := $(SRC)/logbert_embeddings.py
TRANSFORMER_UNSUPERVISED := $(SRC)/transformer_unsupervised_ml.py

# Log types for specific processing
# From src/preprocessing.py --log-type choices
PREPROCESS_LOG_TYPES := web error auth dns firewall vpn audit kernel systemd
# From src/fasttext_embedding.py LOG_TYPE_ATTACKS.keys()
FASTTEXT_LOG_TYPES := dns network web error monitoring auth audit
# Available log types for embeddings (both FastText and LogBERT)
EMBEDDING_LOG_TYPES := dns network web error monitoring auth audit

# Contexts for xgboost_ml.py and ml_models.py (specific logical types + all_combined)
ML_CONTEXTS := all_combined $(FASTTEXT_LOG_TYPES)

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
	@echo "  ml               Run all ML models (all_combined)"
	@echo "  ml-[rf|xgb|knn|lr]     Run specific ML model (all_combined)"
	@echo "  ml-<type>        Run all ML models for specific log type (e.g., make ml-web)"
	@echo "  ml-<type>-[rf|xgb|knn|lr]  Run specific ML model for log type (e.g., make ml-web-rf)"
	@echo "                     Available log types for ml: $(ML_CONTEXTS)"
	@echo "  ml-evaluate      Evaluate without training"
	@echo "  xgboost-ml         Run XGBoost Multi-Label (OvR) for all_combined context"
	@echo "  xgboost-ml-<ctx>   Run XGBoost Multi-Label (OvR) for a specific context"
	@echo "                     Available contexts for xgboost-ml: $(ML_CONTEXTS)"
	@echo "  logbert          Run LogBERT embeddings extraction (all log types, FastText-compatible format)"
	@echo "  logbert-<type>   Run LogBERT embeddings for specific log type"
	@echo "                     Available log types for logbert: $(EMBEDDING_LOG_TYPES)"
	@echo "  transformer-ml   Run transformer-based unsupervised ML (all log types with embeddings)"
	@echo "  transformer-ml-<type>  Run transformer unsupervised ML for specific log type"
	@echo "  transformer-ml-fast    Run transformer unsupervised ML (skip transformer training)"
	@echo "                     Available log types for transformer-ml: $(EMBEDDING_LOG_TYPES)"
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

# ML models with all_combined (default)
ml:         ; $(PYTHON) $(ML)
ml-rf:      ; $(PYTHON) $(ML) --model rf
ml-xgb:     ; $(PYTHON) $(ML) --model xgb
ml-knn:     ; $(PYTHON) $(ML) --model knn
ml-lr:      ; $(PYTHON) $(ML) --model lr

# ML models for specific log type
ml-%:
	@echo "Running all ML models for log type: $*"
	$(PYTHON) $(ML) --log-type $*

# ML models with specific model and log type
ml-%-rf:    ; $(PYTHON) $(ML) --model rf --log-type $*
ml-%-xgb:   ; $(PYTHON) $(ML) --model xgb --log-type $*
ml-%-knn:   ; $(PYTHON) $(ML) --model knn --log-type $*
ml-%-lr:    ; $(PYTHON) $(ML) --model lr --log-type $*

# XGBoost Multi-Label (OvR) baseline
xgboost-ml:
	@echo "Running XGBoost Multi-Label (OvR) for context: all_combined"
	$(PYTHON) $(XGBOOST_ML) --context all_combined

xgboost-ml-%:
	@echo "Running XGBoost Multi-Label (OvR) for context: $*"
	$(PYTHON) $(XGBOOST_ML) --context $*

# LogBERT Embeddings (FastText-compatible format)
logbert:
	@echo "Extracting LogBERT CLS embeddings (all log types, FastText-compatible format)"
	$(PYTHON) $(LOGBERT_EMBEDDINGS)

logbert-%:
	@echo "Extracting LogBERT CLS embeddings for log type: $*"
	$(PYTHON) $(LOGBERT_EMBEDDINGS) --log-type $*

# Transformer Unsupervised ML
transformer-ml:
	@echo "Running transformer-based unsupervised ML (all log types with embeddings)"
	$(PYTHON) $(TRANSFORMER_UNSUPERVISED)

transformer-ml-%:
	@echo "Running transformer-based unsupervised ML for log type: $*"
	$(PYTHON) $(TRANSFORMER_UNSUPERVISED) --log-type $*

transformer-ml-fast:
	@echo "Running transformer-based unsupervised ML (skip transformer training)"
	$(PYTHON) $(TRANSFORMER_UNSUPERVISED) --skip-training

ml-evaluate: ; $(PYTHON) $(ML) --no-train
gan:         ; $(PYTHON) $(GAN)
eval:        ; $(PYTHON) $(EVAL)

# VAE-GAN pipeline
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
        ml-evaluate xgboost-ml logbert transformer-ml transformer-ml-fast gan eval gan-pipeline gan-small eval-fast clean clean-aug clean-all help \
        $(addprefix ml-,$(ML_CONTEXTS)) \
        $(foreach ctx,$(ML_CONTEXTS),$(addprefix ml-$(ctx)-,rf xgb knn lr)) \
        $(addprefix xgboost-ml-,$(ML_CONTEXTS)) \
        $(addprefix logbert-,$(EMBEDDING_LOG_TYPES)) \
        $(addprefix transformer-ml-,$(EMBEDDING_LOG_TYPES))
