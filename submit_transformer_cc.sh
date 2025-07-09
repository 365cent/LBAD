#!/bin/bash
#SBATCH --job-name=transformer_ml
#SBATCH --account=def-youraccountname  # Replace with your Compute Canada account
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4              # Request 4 H100 GPUs
#SBATCH --cpus-per-task=32             # 8 CPUs per GPU
#SBATCH --mem=200G                     # Request 200GB RAM
#SBATCH --output=transformer_%j.out
#SBATCH --error=transformer_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com  # Replace with your email

# Load required modules (adjust based on your cluster)
module load python/3.11
module load cuda/12.1
module load cudnn/8.9.0
module load scipy-stack/2023b

# Create and activate virtual environment
if [ ! -d "venv_transformer" ]; then
    echo "Creating virtual environment..."
    python -m venv venv_transformer
fi

source venv_transformer/bin/activate

# Install/upgrade required packages
echo "Installing/upgrading packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers scikit-learn pandas numpy matplotlib seaborn
pip install psutil halo

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=INFO

# Set distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4
export RANK=0

# Create necessary directories
mkdir -p results models logs

# Print system information
echo "=== System Information ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU Info:"
nvidia-smi
echo "CPU Info:"
lscpu | grep "Model name"
echo "Memory Info:"
free -h
echo "=========================="

# Function to run distributed training
run_distributed() {
    echo "Starting distributed training on 4 GPUs..."
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port=12355 \
        src/transformer.py \
        > logs/distributed_training.log 2>&1
}

# Function to run single GPU training
run_single() {
    echo "Starting single GPU training..."
    python src/transformer.py > logs/single_training.log 2>&1
}

# Check if multiple GPUs are available and run accordingly
if [ $(nvidia-smi -L | wc -l) -gt 1 ]; then
    echo "Multiple GPUs detected, using distributed training"
    run_distributed
else
    echo "Single GPU detected, using single GPU training"
    run_single
fi

# Capture exit status
exit_status=$?

# Create summary report
echo "=== Job Summary ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Exit Status: $exit_status"
echo "=================="

# Archive results
if [ $exit_status -eq 0 ]; then
    echo "Job completed successfully. Archiving results..."
    tar -czf results_${SLURM_JOB_ID}.tar.gz results/ models/ logs/
    echo "Results archived to results_${SLURM_JOB_ID}.tar.gz"
else
    echo "Job failed with exit status $exit_status"
    echo "Check logs for details"
fi

# Clean up if successful
if [ $exit_status -eq 0 ] && [ "${CLEANUP_ON_SUCCESS:-false}" = "true" ]; then
    echo "Cleaning up temporary files..."
    rm -rf venv_transformer
fi

exit $exit_status