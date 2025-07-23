# LBAD - Log-Based Anomaly Detection Pipeline
# ============================================

# Variables
PYTHON := python
PIP := pip
SRC := ./src

# Available log types (check embeddings/ directory for actual availability)
LOG_TYPES := wp-access wp-error dns network web error monitoring auth audit

# =============================================================================
# Main Pipeline Commands
# =============================================================================

help:
	@echo "LBAD - Log-Based Anomaly Detection Pipeline"
	@echo "==========================================="
	@echo ""
	@echo "🚀 MAIN PIPELINES:"
	@echo "  pipeline-all        Complete pipeline: preprocess → embeddings → train → evaluate"
	@echo "  pipeline-<type>     Run full pipeline for specific log type (e.g. pipeline-wp-error)"
	@echo "  pipeline-eval       Quick evaluation pipeline (skip training, use existing models)"
	@echo ""
	@echo "📊 DATA PREPARATION:"
	@echo "  preprocess          Preprocess all log types"
	@echo "  preprocess-<type>   Preprocess specific log type"
	@echo "  embeddings          Generate all embeddings (LogBERT + FastText)"
	@echo "  embeddings-<type>   Generate embeddings for specific log type"
	@echo ""
	@echo "🤖 MODEL TRAINING:"
	@echo "  train               Train transformer models for all log types"
	@echo "  train-<type>        Train transformer for specific log type"
	@echo "  train-sample        Train with sample data (faster, for testing)"
	@echo ""
	@echo "📈 EVALUATION:"
	@echo "  evaluate            Evaluate all trained models"
	@echo "  evaluate-<type>     Evaluate specific log type"
	@echo "  compare             Compare transformer vs f-AnoGAN models"
	@echo "  compare-<type>      Compare models for specific log type"
	@echo ""
	@echo "🔧 STANDALONE TOOLS:"
	@echo "  ml-baseline         Run traditional ML baselines (RF, XGBoost, etc.)"
	@echo "  gan                 Run f-AnoGAN training and evaluation"
	@echo "  view-data           View processed data and embeddings"
	@echo "  test                Run tests"
	@echo ""
	@echo "🧹 MAINTENANCE:"
	@echo "  install             Install dependencies"
	@echo "  clean               Remove cache files"
	@echo "  clean-models        Remove trained models"
	@echo "  clean-all           Remove all generated data"
	@echo ""
	@echo "Available log types: $(LOG_TYPES)"

# =============================================================================
# Complete Pipelines
# =============================================================================

# Full pipeline for all log types
pipeline-all: preprocess embeddings train evaluate
	@echo "✅ Complete pipeline finished for all log types"

# Full pipeline for specific log type
pipeline-%:
	@echo "🚀 Running complete pipeline for log type: $*"
	$(MAKE) preprocess-$*
	$(MAKE) embeddings-$*
	$(MAKE) train-$*
	$(MAKE) evaluate-$*
	@echo "✅ Pipeline completed for $*"

# Quick evaluation pipeline (use existing models)
pipeline-eval: evaluate compare
	@echo "✅ Evaluation pipeline completed"

# =============================================================================
# Data Preparation
# =============================================================================

# Preprocessing
preprocess:
	@echo "📊 Preprocessing all log types..."
	$(PYTHON) $(SRC)/preprocessing.py

preprocess-%:
	@echo "📊 Preprocessing log type: $*"
	$(PYTHON) $(SRC)/preprocessing.py --log-type $*

# Embeddings generation
embeddings:
	@echo "🔄 Generating all embeddings..."
	$(MAKE) logbert
	$(MAKE) fasttext

embeddings-%:
	@echo "🔄 Generating embeddings for log type: $*"
	$(MAKE) logbert-$*
	$(MAKE) fasttext-$*

# LogBERT embeddings
logbert:
	@echo "🧠 Generating LogBERT embeddings for all log types..."
	$(PYTHON) $(SRC)/logbert_embeddings.py

logbert-%:
	@echo "🧠 Generating LogBERT embeddings for log type: $*"
	$(PYTHON) $(SRC)/logbert_embeddings.py --log-type $*

# FastText embeddings
fasttext:
	@echo "📝 Generating FastText embeddings for all log types..."
	$(PYTHON) $(SRC)/fasttext_embedding.py

fasttext-%:
	@echo "📝 Generating FastText embeddings for log type: $*"
	$(PYTHON) $(SRC)/fasttext_embedding.py --log-type $*

# =============================================================================
# Model Training
# =============================================================================

# Train transformer models
train:
	@echo "🤖 Training transformer models for all log types..."
	$(PYTHON) $(SRC)/transformer.py

train-%:
	@echo "🤖 Training transformer model for log type: $*"
	$(PYTHON) $(SRC)/transformer.py --log-type $*

# Train with sample data (faster)
train-sample:
	@echo "🤖 Training transformer models with sample data..."
	$(PYTHON) $(SRC)/transformer.py --sample-size 5000

train-%-sample:
	@echo "🤖 Training transformer model for $* with sample data..."
	$(PYTHON) $(SRC)/transformer.py --log-type $* --sample-size 5000

# =============================================================================
# Model Evaluation
# =============================================================================

# Evaluate trained models
evaluate:
	@echo "📈 Evaluating all trained models..."
	$(PYTHON) $(SRC)/evaluate_models.py

evaluate-%:
	@echo "📈 Evaluating model for log type: $*"
	$(PYTHON) $(SRC)/evaluate_models.py --log-type $*

# Compare different model types
compare:
	@echo "⚖️  Comparing transformer vs f-AnoGAN models..."
	$(PYTHON) $(SRC)/evaluate_models.py --compare-all

compare-%:
	@echo "⚖️  Comparing models for log type: $*"
	$(PYTHON) $(SRC)/evaluate_models.py --log-type $* --compare

# =============================================================================
# Standalone Tools
# =============================================================================

# Traditional ML baselines
ml-baseline:
	@echo "📊 Running traditional ML baselines..."
	$(PYTHON) $(SRC)/ml_models.py

ml-baseline-%:
	@echo "📊 Running ML baselines for log type: $*"
	$(PYTHON) $(SRC)/ml_models.py --log-type $*

# XGBoost specifically
xgboost:
	@echo "🌲 Running XGBoost models..."
	$(PYTHON) $(SRC)/xgboost_ml.py

xgboost-%:
	@echo "🌲 Running XGBoost for context: $*"
	$(PYTHON) $(SRC)/xgboost_ml.py --context $*

# f-AnoGAN (GAN-based anomaly detection)
gan:
	@echo "🎭 Running f-AnoGAN training and evaluation..."
	$(PYTHON) $(SRC)/gan_augmentation.py
	$(PYTHON) $(SRC)/gan_evaluation.py

gan-train:
	@echo "🎭 Training f-AnoGAN models..."
	$(PYTHON) $(SRC)/gan_augmentation.py

gan-eval:
	@echo "🎭 Evaluating f-AnoGAN models..."
	$(PYTHON) $(SRC)/gan_evaluation.py

# Data viewing and analysis
view-data:
	@echo "👁️  Viewing processed data and embeddings..."
	$(PYTHON) $(SRC)/pickle_viewer.py

view-embeddings:
	@echo "👁️  Viewing embedding statistics..."
	$(PYTHON) $(SRC)/embedding_testing.py

# Testing
test:
	@echo "🧪 Running tests..."
	$(PYTHON) $(SRC)/test_transformer.py
	$(PYTHON) $(SRC)/test_combined_transformer.py

test-preprocess:
	@echo "🧪 Testing preprocessing..."
	$(PYTHON) $(SRC)/preprocess_testing.py

# =============================================================================
# Development and Maintenance
# =============================================================================

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	$(PIP) install -r requirements.txt

# Status check - show what's available
status:
	@echo "📋 LBAD Pipeline Status:"
	@echo "======================="
	@echo "📁 Processed data:"
	@ls -la processed/ 2>/dev/null || echo "   No processed data found"
	@echo ""
	@echo "🔗 Embeddings:"
	@ls -la embeddings/ 2>/dev/null || echo "   No embeddings found"
	@echo ""
	@echo "🤖 Models:"
	@ls -la models/ 2>/dev/null || echo "   No trained models found"
	@echo ""
	@echo "📊 Results:"
	@ls -la results/ 2>/dev/null || echo "   No evaluation results found"

# Cleaning
clean:
	@echo "🧹 Cleaning cache files..."
	rm -rf __pycache__/ *.pyc
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-models:
	@echo "🧹 Removing trained models..."
	rm -rf models/* checkpoints/*
	mkdir -p models checkpoints

clean-results:
	@echo "🧹 Removing evaluation results..."
	rm -rf results/*
	mkdir -p results

clean-embeddings:
	@echo "🧹 Removing embeddings..."
	rm -rf embeddings/*
	mkdir -p embeddings

clean-processed:
	@echo "🧹 Removing processed data..."
	rm -rf processed/*
	mkdir -p processed

clean-all: clean
	@echo "🧹 Removing all generated data..."
	rm -rf processed embeddings models results checkpoints augmented
	mkdir -p processed embeddings models results checkpoints augmented

# =============================================================================
# Quick Start Examples
# =============================================================================

# Quick start for new users
quickstart:
	@echo "🚀 LBAD Quick Start Guide:"
	@echo "========================="
	@echo ""
	@echo "1. Install dependencies:"
	@echo "   make install"
	@echo ""
	@echo "2. Run sample pipeline for wp-error:"
	@echo "   make pipeline-wp-error"
	@echo ""
	@echo "3. Or run full pipeline for all types:"
	@echo "   make pipeline-all"
	@echo ""
	@echo "4. Check results:"
	@echo "   make status"
	@echo ""
	@echo "For help: make help"

# Demo with sample data
demo:
	@echo "🎬 Running demo with sample data..."
	$(MAKE) preprocess-wp-error
	$(MAKE) embeddings-wp-error
	$(MAKE) train-wp-error-sample
	$(MAKE) evaluate-wp-error
	@echo "✅ Demo completed! Check results/ directory"

# =============================================================================
# Advanced Workflows
# =============================================================================

# Research workflow - train and compare all models
research:
	@echo "🔬 Running research workflow..."
	$(MAKE) pipeline-all
	$(MAKE) ml-baseline
	$(MAKE) gan
	$(MAKE) compare
	@echo "✅ Research workflow completed"

# Production workflow - optimized for specific log type
production-%:
	@echo "🏭 Running production workflow for $*..."
	$(MAKE) preprocess-$*
	$(MAKE) logbert-$*
	$(MAKE) train-$*
	$(MAKE) evaluate-$*
	@echo "✅ Production workflow completed for $*"

# Benchmark all models
benchmark:
	@echo "⏱️  Running benchmark of all models..."
	time $(MAKE) train-sample
	time $(MAKE) ml-baseline
	time $(MAKE) gan-train
	$(MAKE) compare
	@echo "✅ Benchmark completed"

.PHONY: help pipeline-all pipeline-eval preprocess embeddings logbert fasttext train evaluate compare \
        ml-baseline xgboost gan gan-train gan-eval view-data view-embeddings test test-preprocess \
        install status clean clean-models clean-results clean-embeddings clean-processed clean-all \
        quickstart demo research benchmark \
        $(addprefix pipeline-,$(LOG_TYPES)) \
        $(addprefix preprocess-,$(LOG_TYPES)) \
        $(addprefix embeddings-,$(LOG_TYPES)) \
        $(addprefix logbert-,$(LOG_TYPES)) \
        $(addprefix fasttext-,$(LOG_TYPES)) \
        $(addprefix train-,$(LOG_TYPES)) \
        $(addprefix train-,$(addsuffix -sample,$(LOG_TYPES))) \
        $(addprefix evaluate-,$(LOG_TYPES)) \
        $(addprefix compare-,$(LOG_TYPES)) \
        $(addprefix ml-baseline-,$(LOG_TYPES)) \
        $(addprefix production-,$(LOG_TYPES))
