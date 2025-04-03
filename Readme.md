# Log-Based Attack Detection (LBAD) Framework

A robust framework for detecting and classifying attacks in system logs using advanced embedding techniques and machine learning models. Optimized for performance on Apple Silicon.

## Overview

LBAD tackles challenges in log-based attack detection by:
- Representing unstructured log data with FastText, Word2Vec, and TF-IDF embeddings.
- Leveraging traditional machine learning models (Random Forest, XGBoost, SVM, KNN, Logistic Regression) for accurate classification.
- Optimizing performance on Apple Silicon (M1/M2/M3) processors.
- Support for GAN-based data augmentation is planned for future releases.

## Features

- **Log Preprocessing**: Tokenization, normalization, and filtering of raw logs.
- **Embedding Generation**: Multiple embedding methods:
  - **FastText**: Capturing subword information for better representation of unusual log entries.
  - **Word2Vec**: Creating semantic vector representations.
  - **TF-IDF**: Weighting terms by importance with dimensionality reduction.
- **Machine Learning Models**: Five optimized classifiers:
  - **Random Forest**: Robust ensemble method for imbalanced datasets.
  - **XGBoost**: Gradient-boosted trees with high performance.
  - **SVM**: Support Vector Machine with non-linear kernels.
  - **KNN**: K-Nearest Neighbors with distance weighting.
  - **Logistic Regression**: Fast linear model with regularization.
- **Comprehensive Evaluation**: Precision, recall, F1-score, confusion matrices, and per-class metrics.
- **Coming Soon**: GAN-based data augmentation for minority classes.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/365cent/LBAD.git
   cd LBAD
   ```

2. Install dependencies:
   ```bash
   make install
   ```

## Usage

### Complete Pipeline

Run the entire workflow:
```bash
make all
```

### Individual Components

- **Preprocessing**:
  ```bash
  make preprocess
  ```

- **Embedding Generation**:
  ```bash
  make embeddings    # Generate all embedding types
  make fasttext      # Generate FastText embeddings
  make word2vec      # Generate Word2Vec embeddings
  make tfidf         # Generate TF-IDF embeddings
  ```

- **Machine Learning**:
  ```bash
  make ml            # Run all ML models on all embeddings
  make ml-fasttext   # ML with FastText embeddings
  make ml-word2vec   # ML with Word2Vec embeddings
  make ml-tfidf      # ML with TF-IDF embeddings
  
  # Individual model types
  make ml-rf         # Random Forest classifier
  make ml-xgb        # XGBoost classifier
  make ml-svm        # SVM classifier
  make ml-knn        # KNN classifier
  make ml-lr         # Logistic Regression classifier
  ```

- **Direct TFRecord Processing**:
  ```bash
  make ml-direct     # Process TFRecord data directly (no embeddings)
  ```

- **Sample Run**:
  ```bash
  make sample-run    # Run with limited samples (for testing)
  ```

### Cleanup

Remove generated files:
```bash
make clean       # Remove cache files
make clean-all   # Remove all generated data
```

## Project Structure

```
.
├── logs/                       # Raw log dataset
├── labels/                     # Ground truth annotations
├── src/                        # Source code
│   ├── preprocessing.py        # Log preprocessing
│   ├── preprocess_testing.py   # Testing for preprocessing
│   ├── fasttext_embedding.py   # FastText embedding generation
│   ├── word2vec_embedding.py   # Word2Vec embedding generation
│   ├── tfidf_embedding.py      # TF-IDF embedding generation
│   ├── embedding_testing.py    # Testing for embeddings
│   ├── ml_models.py            # Traditional ML models
│   ├── gan_augmentation.py.dist # GAN templates (future implementation)
│   ├── gan_evaluation.py.dist  # GAN evaluation templates
│   ├── gan_ml.py.dist          # GAN ML templates
│   └── main.py                 # Workflow orchestration
├── Makefile                    # Build automation
└── requirements.txt            # Dependencies
```

## Evaluation Metrics

- **Precision**: Reduces false positives in attack detection.
- **Recall**: Ensures comprehensive attack detection.
- **F1-Score**: Balances precision and recall.
- **Per-class metrics**: Detailed performance for each attack type.
- **Confusion matrices**: Visual representations of classification performance.

## Advanced Techniques

- **TF-IDF with SVD**: Dimension reduction while preserving signal.
- **Multi-embedding comparison**: Test different vector representations.
- **Auto-detection of processed files**: Automatically runs preprocessing when needed.
- **Direct TFRecord processing**: Option to vectorize logs without embeddings.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this framework in your research, please cite:
```
@misc{log_attack_detection,
  author = {365cent},
  title = {Log-Based Attack Detection Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/365cent/LBAD}
}
```

## Acknowledgments

- **FastText, Word2Vec, and TF-IDF**: For embedding implementations.
- **Scikit-learn, XGBoost**: For ML implementations.
- **TensorFlow**: For data processing frameworks.
- **AIT Log Data Set**: For benchmark datasets.
- **Open-Source Community**: For tools and libraries.
