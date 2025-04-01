# Log-Based Attack Detection (LBAD) framework

This repository provides a framework for detecting and classifying attacks in system logs using FastText embeddings and traditional machine learning algorithms.

## Overview

The framework addresses challenges in log-based attack detection by:
- Representing unstructured log data with FastText embeddings.
- Applying machine learning models (Random Forest, XGBoost, SVM) for attack classification.
- Exploring advanced techniques like GAN-based data augmentation and Granite embeddings.

## Features

- **Log Preprocessing**: Tokenization, normalization, and filtering of raw logs.
- **FastText Embeddings**: Semantic representation of log entries.
- **Machine Learning Models**: Random Forest, XGBoost, and SVM for classification.
- **Optional Enhancements**: GAN-based augmentation and Granite embeddings.
- **Evaluation Metrics**: Precision, recall, F1-score, and ROC-AUC for imbalanced datasets.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/log-attack-detection.git
   cd log-attack-detection
   ```

2. Install dependencies:
   ```bash
   make install
   ```

## Usage

### Complete Pipeline

Run the entire pipeline with a single command:
```bash
make all
```

### Individual Components

- **Preprocessing**: Parse and normalize logs
  ```bash
  make preprocess
  ```

- **Embedding Generation**:
  ```bash
  make embed     # FastText embeddings
  ```

- **Machine Learning**:
  ```bash
  make test      # Run testing script
  ```

### Cleanup

Remove generated files:
```bash
make clean
```

## Project Structure

```
.
├── logs/                   # Raw log from AIT Log Data Set V2.0
├── labels/                 # Labels from AIT Log Data Set V2.0
├── src/                    # Source code
│   ├── preprocessing.py    # Log preprocessing
│   ├── fasttext_embedding.py # FastText embedding generation
│   ├── testing.py          # Testing script
│   └── main.py             # Main workflow orchestration
├── Makefile                # Build automation
└── requirements.txt        # Dependencies
```

## Evaluation Metrics

- **Precision**: Avoid false positives.
- **Recall**: Detect all attack instances.
- **F1-Score**: Balance between precision and recall.
- **ROC-AUC**: Distinguish between normal and attack classes.

## Advanced Techniques

- **GAN-Based Augmentation**: Address class imbalance by generating synthetic log entries.
- **Granite Embeddings**: Capture structural relationships in log data for complex attack patterns.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this framework in your research, please cite:
```
@misc{log_attack_detection,
  author = {Your Name},
  title = {Log-Based Attack Detection Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/log-attack-detection}
}
```

## Acknowledgments

- **FastText**: For embedding implementation.
- **AIT Log Data Set**: For benchmark datasets.
- **Open-Source Community**: For tools and libraries.
