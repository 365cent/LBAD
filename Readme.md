# Log-Based Attack Detection (LBAD) Framework

A robust framework for detecting and classifying attacks in system logs using advanced embedding techniques and machine learning models.

## Overview

LBAD tackles challenges in log-based attack detection by:
- Representing unstructured log data with FastText and Word2Vec embeddings.
- Leveraging machine learning models (Random Forest, XGBoost, SVM) for accurate classification.
- Mitigating class imbalance with GAN-based data augmentation.

## Features

- **Log Preprocessing**: Tokenization, normalization, and filtering of raw logs.
- **Embedding Generation**: FastText and Word2Vec for semantic vector representations.
- **Machine Learning Models**: Random Forest, XGBoost, and SVM optimized for imbalanced datasets.
- **Data Augmentation**: GAN-based synthetic data generation.
- **Evaluation Metrics**: Precision, recall, F1-score, and ROC-AUC.

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
  make embed     # FastText embeddings
  make word2vec  # Word2Vec embeddings
  ```

- **Machine Learning**:
  ```bash
  make ml        # Run all ML models
  make ml-rf     # Random Forest classifier
  make ml-xgb    # XGBoost classifier
  make ml-svm    # SVM classifier
  ```

- **Data Augmentation**:
  ```bash
  make gan
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
│   ├── embedding_testing.py    # Testing for embeddings
│   ├── ml_models.py            # Machine learning models
│   ├── gan_augmentation.py     # GAN-based data augmentation
│   ├── granite_embedding.py.dist # Granite embedding implementation (placeholder)
│   └── main.py                 # Workflow orchestration
├── Makefile                    # Build automation
└── requirements.txt            # Dependencies
```

## Evaluation Metrics

- **Precision**: Reduces false positives.
- **Recall**: Ensures comprehensive attack detection.
- **F1-Score**: Balances precision and recall.
- **ROC-AUC**: Measures classification performance across thresholds.

## Advanced Techniques

- **GAN Augmentation**: Generates synthetic data to address class imbalance.
- **Embedding Optimization**: FastText and Word2Vec for efficient log representation.

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

- **FastText & Word2Vec**: For embedding implementations.
- **AIT Log Data Set**: For benchmark datasets.
- **Open-Source Community**: For tools and libraries.
