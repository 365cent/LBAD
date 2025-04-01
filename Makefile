# Makefile for Python Project

# Variables
PYTHON := python
PIP := pip
PREPROCESSING := ./src/preprocessing.py
TESTING := ./src/testing.py
FASTTEXT := ./src/fasttext_embedding.py
WORD2VEC := ./src/word2vec_embedding.py
ML := ./src/ml_models.py
MAIN := ./src/main.py
GAN := ./src/gan_augmentation.py
REQUIREMENTS := requirements.txt

# Help command
help:
	@echo "Available targets:"
	@echo "  all         - Run preprocessing, testing, embedding, and machine learning models"
	@echo "  install     - Install dependencies in the current Python environment"
	@echo "  preprocess  - Run the preprocessing script"
	@echo "  test        - Run the testing script"
	@echo "  embed       - Run the fastText embedding script"
	@echo "  word2vec    - Run the Word2Vec embedding script"
	@echo "  ml          - Run machine learning models"
	@echo "  ml-rf       - Run the machine learning model with random forest"
	@echo "  ml-xgb      - Run the machine learning model with XGBoost"
	@echo "  ml-svm      - Run the machine learning model with SVM"
	@echo "  ml-evaluate - Evaluate the machine learning model (without training)"
	@echo "  gan         - Run GAN-based data augmentation"
	@echo "  clean       - Clean up generated files"
	@echo "  help        - Display this help message"

# Run all scripts in order
all:
	$(PYTHON) $(MAIN)

# Install dependencies in the current Python environment
install:
	$(PIP) install -r $(REQUIREMENTS)

# Run the preprocessing script
preprocess:
	$(PYTHON) $(PREPROCESSING)

# Run the testing script
test:
	$(PYTHON) $(TESTING)

# Run the fastText embedding script
embed:
	$(PYTHON) $(FASTTEXT)

# Run the Word2Vec embedding script
word2vec:
	$(PYTHON) $(WORD2VEC)

# Run machine learning models
ml:
	$(PYTHON) $(ML)

# Run the machine learning model with random forest
ml-rf:
	$(PYTHON) $(ML) --model rf

# Run the machine learning model with XGBoost
ml-xgb:
	$(PYTHON) $(ML) --model xgb

# Run the machine learning model with SVM
ml-svm:
	$(PYTHON) $(ML) --model svm

# Evaluate the machine learning model (without training)
ml-evaluate:
	$(PYTHON) $(ML) --evaluate-only

# Run GAN-based data augmentation
gan:
	$(PYTHON) $(GAN)

# Clean up generated files
clean:
	rm -rf __pycache__/
	rm -rf *.pyc
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Clean all generated files including embeddings, models, and results, then run clean, use with caution
clean-all:
	rm -rf processed
	rm -rf embedding
	rm -rf models
	rm -rf results
	rm -rf augmentation
	$(MAKE) clean

.PHONY: all install preprocess test embed word2vec ml ml-rf ml-xgb ml-svm ml-evaluate gan clean help clean-all
