# Log-Based Attack Detection (LBAD) Framework

A robust framework for detecting and classifying attacks in system logs using advanced embedding techniques, machine learning models, and VAE-GAN augmentation. Optimized for performance on Apple Silicon.

## Overview

LBAD tackles challenges in log-based attack detection by:
- Representing unstructured log data with FastText, Word2Vec, and TF-IDF embeddings.
- Leveraging traditional machine learning models (Random Forest, XGBoost, SVM, KNN, Logistic Regression) for accurate classification.
- Implementing VAE-GAN (Variational Autoencoder - Generative Adversarial Network) hybrid for high-quality synthetic data generation.
- Addressing class imbalance through advanced data augmentation techniques.
- Optimizing performance on Apple Silicon (M1/M2/M3) processors.

## Pipeline Architecture

```mermaid
flowchart TD
    subgraph Log_Preprocessing
        A[Scan and filter text log files] --> B{Match logs with label files?};
        B -- Yes --> C[Generate TensorFlow Examples];
        C --> D_TF[Serialize and store TFRecord files by log type e.g., processed/web/*.tfrecord];
        B -- No --> D_TF;
    end

    subgraph Embedding_Generation
        direction TB
        D_TF -- Run fasttext_embedding.py --> E_FT_Combined[Store Combined Embeddings & Labels e.g., embeddings/embeddings_all_combined.pkl];
        D_TF -- Run fasttext_embedding.py --log-type &lt;type&gt; --> E_FT_PerType[Store Per-Type Embeddings & Labels e.g., embeddings/web/embeddings.pkl];
        D_TF -- Potentially other embedding scripts --> E_OtherEmbed[Other Embedding Types e.g., Word2Vec, TF-IDF];
    end

    subgraph Baseline_MultiLabel_XGBoost_Evaluation
        direction TB
        E_FT_Combined -- Run xgboost_ml.py --context all_combined --> F_XGB_Combined[Train OvR XGBoost on Combined Data];
        F_XGB_Combined --> G_XGB_Combined_Eval[Evaluate & Store Metrics for Combined];
        
        E_FT_PerType -- Run xgboost_ml.py --context &lt;type&gt; --> F_XGB_PerType[Train OvR XGBoost on Per-Type Data e.g., web];
        F_XGB_PerType --> G_XGB_PerType_Eval[Evaluate & Store Metrics for Per-Type];
    end

    subgraph Traditional_MultiClass_ML_Evaluation
        direction TB
        note_trad["ml_models.py: Handles various embeddings (FastText, W2V, TF-IDF) for multi-class tasks (e.g., first label)"]
        E_FT_Combined ----> H_Trad_FT[Load FastText for ml_models.py];
        E_OtherEmbed ----> H_Trad_Other[Load Other Embeddings for ml_models.py];
        H_Trad_FT --> I_Trad_ML[Train RF, XGB, KNN, LR etc.];
        H_Trad_Other --> I_Trad_ML;
        I_Trad_ML --> J_Trad_Eval[Evaluate & Store Metrics];
    end

    subgraph Optional_VAE_GAN_Data_Augmentation
        direction TB
        note_gan["GAN operates on selected embeddings (e.g., minority classes from FastText combined set)"]
        E_FT_Combined --> K_GAN[Identify minority classes from embeddings];
        K_GAN --> L_GAN[Build & Train VAE-GAN model];
        L_GAN --> M_GAN[Generate synthetic embeddings];
        M_GAN --> N_GAN[Save augmented dataset];
        N_GAN --> O_GAN_Eval[Load Original & Augmented Data for Traditional ML Evaluation e.g. using ml_models.py];
        O_GAN_Eval --> P_GAN_Results[Evaluate & Store Post-Augmentation Metrics];
    end

    style A fill:#lightgreen,stroke:#333,stroke-width:2px
    style B fill:#lightgreen,stroke:#333,stroke-width:2px
    style C fill:#lightgreen,stroke:#333,stroke-width:2px
    style D_TF fill:#lightgreen,stroke:#333,stroke-width:2px

    style E_FT_Combined fill:#lightblue,stroke:#333,stroke-width:2px
    style E_FT_PerType fill:#lightblue,stroke:#333,stroke-width:2px
    style E_OtherEmbed fill:#lightblue,stroke:#333,stroke-width:2px

    style F_XGB_Combined fill:#FFD700,stroke:#333,stroke-width:2px
    style G_XGB_Combined_Eval fill:#FFD700,stroke:#333,stroke-width:2px
    style F_XGB_PerType fill:#FFD700,stroke:#333,stroke-width:2px
    style G_XGB_PerType_Eval fill:#FFD700,stroke:#333,stroke-width:2px
    
    style H_Trad_FT fill:#FFBF00,stroke:#333,stroke-width:2px
    style H_Trad_Other fill:#FFBF00,stroke:#333,stroke-width:2px
    style I_Trad_ML fill:#FFBF00,stroke:#333,stroke-width:2px
    style J_Trad_Eval fill:#FFBF00,stroke:#333,stroke-width:2px

    style K_GAN fill:#FFC0CB,stroke:#333,stroke-width:2px
    style L_GAN fill:#FFC0CB,stroke:#333,stroke-width:2px
    style M_GAN fill:#FFC0CB,stroke:#333,stroke-width:2px
    style N_GAN fill:#FFC0CB,stroke:#333,stroke-width:2px
    style O_GAN_Eval fill:#FFC0CB,stroke:#333,stroke-width:2px
    style P_GAN_Results fill:#FFC0CB,stroke:#333,stroke-width:2px
```

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
- **VAE-GAN Augmentation**: Hybrid approach for generating high-quality synthetic data:
  - **Beta-VAE**: Tunable KL divergence weighting for better representation learning
  - **KL Annealing**: Gradual increase in KL weight for stable training
  - **Multi-batch diversity generation**: Creates embeddings with varying diversity levels
  - **Focused minority class augmentation**: Targeted synthetic data generation
- **Comprehensive Evaluation**: Precision, recall, F1-score, confusion matrices, and per-class metrics.

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

- **VAE-GAN Augmentation and Evaluation**:
  ```bash
  make gan           # Run VAE-GAN augmentation
  make eval          # Evaluate augmentation results
  make gan-pipeline  # Run both augmentation and evaluation
  make gan-small     # Run with fewer epochs and samples (quick test)
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
make clean-aug   # Remove augmented data only
make clean-all   # Remove all generated data
```

## Project Structure

```
.
├── logs/                     # Raw log dataset
├── labels/                   # Ground truth annotations
├── embeddings/               # Generated embeddings
├── augmented/                # Synthetic data from VAE-GAN
├── results/                  # Evaluation metrics and visualizations
├── models/                   # Saved model weights
├── src/                      # Source code
│   ├── preprocessing.py      # Log preprocessing
│   ├── fasttext_embedding.py # FastText embedding generation
│   ├── word2vec_embedding.py # Word2Vec embedding generation
│   ├── tfidf_embedding.py    # TF-IDF embedding generation
│   ├── gan_augmentation.py   # VAE-GAN implementation
│   ├── gan_evaluation.py     # Evaluation with XGBoost
│   ├── ml_models.py          # Traditional ML models
│   └── main.py               # Workflow orchestration
├── Makefile                  # Build automation
└── requirements.txt          # Dependencies
```

## VAE-GAN Augmentation Approach

Our VAE-GAN hybrid approach combines the benefits of Variational Autoencoders and Generative Adversarial Networks to generate high-quality synthetic embeddings:

### Model Architecture
- **Enhanced Encoder**: Multi-layer network (512→768→512 units) for robust feature extraction
- **Latent Space**: 128-dimensional latent space for expressive representation
- **Decoder**: Mirror of encoder with tanh activation for embedding generation
- **Discriminator**: Deep network that differentiates real from synthetic embeddings

### Training Dynamics
- **Beta-VAE**: Reduced beta parameter (0.5) to balance reconstruction vs. regularization
- **KL Annealing**: Gradual increase of KL weight during training for stability
- **Learning Rate Scheduling**: Exponential decay to prevent training divergence
- **Gradient Clipping**: Limits gradient norms to prevent exploding gradients

### Diversity Enhancement
- **Multi-batch Generation**: Creates synthetic samples with varying diversity levels
- **Periodic Diversity Injection**: Mixes diverse samples into training batches
- **Targeted Augmentation**: Focuses on minority attack classes to address imbalance

## Evaluation Metrics

- **Precision**: Reduces false positives in attack detection.
- **Recall**: Ensures comprehensive attack detection.
- **F1-Score**: Balances precision and recall.
- **Per-class metrics**: Detailed performance for each attack type.
- **Confusion matrices**: Visual representations of classification performance.

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
- **TensorFlow, Keras**: For deep learning frameworks.
- **Scikit-learn, XGBoost**: For ML implementations.
- **Open-Source Community**: For tools and libraries.
