# LBAD - Log-Based Anomaly Detection Pipeline
# ============================================

# Variables
PYTHON := python
PIP := pip
SRC := ./src

# Available log types (override at runtime as needed)
LOG_TYPES := wp-access wp-error dns monitor share vpn audit auth

# Local HF/transformers/torch/matplotlib cache to avoid $HOME/.cache quota issues
HF_ENV := HF_HOME="$(PWD)/hf_cache" TRANSFORMERS_CACHE="$(PWD)/hf_cache" HUGGINGFACE_HUB_CACHE="$(PWD)/hf_cache" XDG_CACHE_HOME="$(PWD)/.cache" TORCH_HOME="$(PWD)/.cache/torch" MPLCONFIGDIR="$(PWD)/.mplconfig"

# Setup cache directories (alias for ensure-cache)
setup-cache: ensure-cache

# =============================================================================
# Main Pipeline Commands
# =============================================================================

help:
	@echo "LBAD - Log-Based Anomaly Detection Pipeline"
	@echo "==========================================="
	@echo ""
	@echo "🧪 Orchestration:"
	@echo "  all                Preprocess → generate embeddings → train → evaluate"
	@echo "  vars (override): LOG_TYPES=..."
	@echo ""
	@echo "🚀 MAIN PIPELINES:"
	@echo "  pipeline-all        Complete pipeline: preprocess → embeddings → train → evaluate"
	@echo "  pipeline-<type>     Run full pipeline for specific log type (e.g. pipeline-wp-error)"
	@echo "  pipeline-eval       Quick evaluation pipeline (skip training, use existing models)"
	@echo ""
	@echo "📊 DATA PREPARATION:"
	@echo "  preprocess          Preprocess all log types"
	@echo "  preprocess-<type>   Preprocess specific log type"
	@echo "  embeddings          Generate all embeddings (LogBERT + FastText + Word2Vec)"
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
	@echo "  compare             Compare transformer models"
	@echo "  compare-<type>      Compare models for specific log type"
	@echo ""
	@echo "🔧 STANDALONE TOOLS:"
	@echo "  ml-baseline         Run traditional ML baselines (RF, XGBoost, etc.)"
	@echo "  test                Run tests"
	@echo ""
	@echo "🧹 MAINTENANCE:"
	@echo "  install             Install dependencies"
	@echo "  clean               Remove cache files"
	@echo "  clean-models        Remove trained models"
	@echo "  clean-results       Remove all evaluation results"
	@echo "  clean-old-results   Remove duplicate old results (keep latest)"
	@echo "  clean-all           Remove all generated data"
	@echo ""
	@echo "Available log types: $(LOG_TYPES)"

# =============================================================================
# Complete Pipelines
# =============================================================================

# Default full pipeline target
all: pipeline-all

# Full pipeline for all log types
pipeline-all: setup-cache preprocess embeddings train evaluate
	@echo "✅ Complete pipeline finished for all log types"

# Full pipeline for specific log type
pipeline-%: setup-cache
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
preprocess: ensure-cache
	@echo "📊 Preprocessing all log types..."
	$(HF_ENV) $(PYTHON) $(SRC)/preprocessing.py

preprocess-%: ensure-cache
	@echo "📊 Preprocessing log type: $*"
	$(HF_ENV) $(PYTHON) $(SRC)/preprocessing.py --log-type $*

# Embeddings generation
embeddings:
	@echo "🔄 Generating all embeddings..."
	$(MAKE) logbert
	$(MAKE) fasttext
	$(MAKE) word2vec

embeddings-%:
	@echo "🔄 Generating embeddings for log type: $*"
	$(MAKE) logbert-$*
	$(MAKE) fasttext-$*
	$(MAKE) word2vec-$*

# LogBERT embeddings (write to embeddings/logbert/<type>)
logbert: ensure-cache
	@echo "🧠 Generating LogBERT embeddings for all log types..."
	$(HF_ENV) $(PYTHON) $(SRC)/logbert_embeddings.py --output-subdir logbert

logbert-%: ensure-cache
	@echo "🧠 Generating LogBERT embeddings for log type: $*"
	$(HF_ENV) $(PYTHON) $(SRC)/logbert_embeddings.py --log-type $* --output-subdir logbert

# FastText embeddings (write to embeddings/fasttext/<type>)
fasttext:
	@echo "📝 Generating FastText embeddings for all log types..."
	$(PYTHON) $(SRC)/fasttext_embedding.py --output-subdir fasttext

fasttext-%:
	@echo "📝 Generating FastText embeddings for log type: $*"
	$(PYTHON) $(SRC)/fasttext_embedding.py --log-type $* --output-subdir fasttext

# Word2Vec embeddings (write to embeddings/word2vec/<type>)
word2vec:
	@echo "🔤 Generating Word2Vec embeddings for all log types..."
	$(PYTHON) $(SRC)/word2vec_embedding_thesis.py --output-subdir word2vec

word2vec-%:
	@echo "🔤 Generating Word2Vec embeddings for log type: $*"
	$(PYTHON) $(SRC)/word2vec_embedding_thesis.py --log-type $* --output-subdir word2vec

# =============================================================================
# Model Training
# =============================================================================

# Train transformer models
train: setup-cache
	@echo "🤖 Training transformer models for all log types..."
	$(HF_ENV) $(PYTHON) $(SRC)/transformer.py

train-%: setup-cache
	@echo "🤖 Training transformer model for log type: $*"
	$(HF_ENV) $(PYTHON) $(SRC)/transformer.py --log-type $*

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

# Testing
test:
	@echo "🧪 Running tests..."
	@echo "⚠️  No test suite currently configured"

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

clean-old-results:
	@echo "🧹 Cleaning old duplicate results (keeping latest per combination)..."
	@find results -name "*.txt" -type f | \
		sed 's/_[0-9]\{8\}-[0-9]\{6\}\.txt$$//' | \
		sort | uniq -d | \
		while read prefix; do \
			ls -t $$prefix_*.txt 2>/dev/null | tail -n +2 | xargs rm -f 2>/dev/null || true; \
		done
	@echo "✅ Old results cleaned"

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
# Cache Utilities
# =============================================================================

ensure-cache:
	@mkdir -p hf_cache .cache/torch .mplconfig gensim_data
	@echo "✅ Cache directories ensured in project root"

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

.PHONY: all help pipeline-all pipeline-eval preprocess embeddings logbert fasttext word2vec train evaluate compare \
        ml-baseline test install status clean clean-models clean-results clean-old-results clean-embeddings clean-processed clean-all \
        quickstart demo research \
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
