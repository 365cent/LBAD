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
	@echo "  ml-direct     - Run ML directly on TFRecord data (no embeddings)"
	@echo "  ml-rf         - Run Random Forest model on default embeddings"
	@echo "  ml-xgb        - Run XGBoost model on default embeddings"
	@echo "  ml-svm        - Run SVM model on default embeddings"
	@echo "  ml-knn        - Run KNN model on default embeddings"
	@echo "  ml-lr         - Run Logistic Regression model on default embeddings"
	@echo "  evaluate      - Evaluate GAN augmentation effectiveness"
	@echo "  evaluate-*    - Evaluate using specific model/data combinations"
	@echo "  sample-run    - Run a sample pipeline on limited data"
	@echo "  run-full-with-skip - Run full pipeline skipping errors"
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

ml-direct:
	$(PYTHON) $(MAIN) --ml --direct

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

# GAN Evaluation
evaluate:
	$(PYTHON) $(MAIN) --evaluate

evaluate-rf-original:
	$(PYTHON) $(MAIN) --evaluate --eval-model rf --eval-data original

evaluate-xgb-augmented:
	$(PYTHON) $(MAIN) --evaluate --eval-model xgb --eval-data augmented

# Run with limited samples
sample-run:
	$(PYTHON) $(MAIN) --ml --direct --max-samples 10000

# Run full pipeline and skip errors
run-full-with-skip:
	$(PYTHON) $(MAIN) --all --skip-errors

# Cleanup
clean:
	rm -rf __pycache__/
	rm -rf *.pyc
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

clean-all:
	rm -rf processed embeddings models results augmented evaluation
	$(MAKE) clean

.PHONY: all install preprocess test embeddings fasttext word2vec tfidf ml ml-fasttext ml-word2vec ml-tfidf ml-direct ml-rf ml-xgb ml-svm ml-knn ml-lr evaluate evaluate-rf-original evaluate-xgb-augmented sample-run run-full-with-skip clean clean-all help
