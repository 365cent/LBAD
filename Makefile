# Makefile for Log Analysis Project

# Variables
PYTHON := python3
PIP := pip3
MAIN := ./src/main.py
REQUIREMENTS := requirements.txt

# Help command
help:
	@echo "Available targets:"
	@echo "  all           - Run complete pipeline (preprocessing, all embeddings, ML models)"
	@echo "  install       - Install dependencies in the current Python environment"
	@echo "  preprocess    - Run log preprocessing"
	@echo "  test          - Run preprocessing testing utilities"
	@echo "  embeddings    - Generate all embedding types (FastText, Word2Vec, TF-IDF)"
	@echo "  fasttext      - Generate FastText embeddings"
	@echo "  word2vec      - Generate Word2Vec embeddings"
	@echo "  tfidf         - Generate TF-IDF embeddings"
	@echo "  ml            - Run all ML models on all available embeddings"
	@echo "  ml-fasttext   - Run ML models with FastText embeddings"
	@echo "  ml-word2vec   - Run ML models with Word2Vec embeddings"
	@echo "  ml-tfidf      - Run ML models with TF-IDF embeddings"
	@echo "  ml-rf         - Run Random Forest model on default embeddings"
	@echo "  ml-xgb        - Run XGBoost model on default embeddings"
	@echo "  ml-svm        - Run SVM model on default embeddings"
	@echo "  ml-knn        - Run KNN model on default embeddings"
	@echo "  ml-lr         - Run Logistic Regression model on default embeddings"
	@echo "  ml-no-svm     - Run all models except SVM (faster processing)"
	@echo "  highperf      - Run optimized high-performance pipeline with system optimizations"
	@echo "  run-full-with-skip - Run full pipeline skipping errors"
	@echo "  optimize-system - Apply system optimizations for ML workloads"
	@echo "  clean         - Clean up temporary files and caches"
	@echo "  clean-all     - Clean all generated files (use with caution)"
	@echo "  help          - Display this help message"

# Run full pipeline
all:
	$(PYTHON) $(MAIN) --all

# Install dependencies
install:
	$(PIP) install -r $(REQUIREMENTS)

# Data preprocessing
preprocess:
	$(PYTHON) $(MAIN) --preprocess

# Testing utilities
test:
	$(PYTHON) $(MAIN) --test

# All embeddings
embeddings: fasttext word2vec tfidf

fasttext:
	$(PYTHON) $(MAIN) --fasttext

word2vec:
	$(PYTHON) $(MAIN) --word2vec

tfidf:
	$(PYTHON) $(MAIN) --tfidf

# Machine Learning models
ml:
	$(PYTHON) $(MAIN) --ml --all-embeddings

ml-fasttext:
	$(PYTHON) $(MAIN) --ml --fasttext

ml-word2vec:
	$(PYTHON) $(MAIN) --ml --word2vec

ml-tfidf:
	$(PYTHON) $(MAIN) --ml --tfidf

ml-rf:
	$(PYTHON) $(MAIN) --ml --model rf

ml-xgb:
	$(PYTHON) $(MAIN) --ml --model xgb

ml-svm:
	$(PYTHON) $(MAIN) --ml --model svm

ml-knn:
	$(PYTHON) $(MAIN) --ml --model knn

ml-lr:
	$(PYTHON) $(MAIN) --ml --model lr

# Run all models except SVM (faster processing)
ml-no-svm:
	$(PYTHON) $(MAIN) --ml --model rf,xgb,knn,lr

# High-performance run with system optimizations
highperf:
	@echo "Running high-performance ML pipeline with system optimizations..."
	@sudo purge
	@sudo nice -n -20 $(PYTHON) $(MAIN) --all

# Run full pipeline and skip errors
run-full-with-skip:
	$(PYTHON) $(MAIN) --all --skip-errors

# System optimization script
optimize-system:
	@echo "Applying system optimizations for ML workloads..."
	@sudo purge
	@echo "Memory cache cleared"
	@sudo killall mds_stores || true
	@echo "Spotlight indexing temporarily paused"
	@echo "System optimized for ML workloads"

# Cleanup
clean:
	rm -rf __pycache__/
	rm -rf *.pyc
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

clean-all:
	rm -rf processed embeddings models results augmented evaluation
	$(MAKE) clean

.PHONY: all install preprocess test embeddings fasttext word2vec tfidf ml ml-fasttext ml-word2vec ml-tfidf ml-rf ml-xgb ml-svm ml-knn ml-lr ml-no-svm highperf run-full-with-skip optimize-system clean clean-all help
