# High-Performance Transformer Training on Compute Canada

This repository contains an optimized unsupervised multi-label transformer implementation specifically designed for Compute Canada's supercomputing clusters (Cedar, Graham, Beluga, Narval).

## 🚀 Key Features

- **Auto-Resource Detection**: Automatically detects and optimizes for H100, A100, V100, and other GPUs
- **Multi-GPU Support**: Full distributed training across multiple nodes and GPUs
- **Memory Optimization**: Intelligent chunked processing for large datasets (100M+ samples)
- **Cluster-Specific Tuning**: Optimized configurations for each Compute Canada cluster
- **Mixed Precision Training**: Automatic FP16/BF16 support for faster training
- **Comprehensive Monitoring**: Real-time GPU, memory, and training metrics
- **Fault Tolerance**: Automatic checkpointing and recovery mechanisms
- **File Output**: All step results automatically saved to files

## 📋 Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 32+ GB RAM (recommended)
- PyTorch 2.0+

## 🛠 Quick Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd LBAD

# Run automated setup
python setup_compute_canada.py --account your_cc_account --email your.email@university.ca
```

### 2. Manual Setup (Alternative)
```bash
# Load modules (adjust for your cluster)
module load python/3.11 cuda/12.1 cudnn/8.9.0

# Create virtual environment
python -m venv venv_transformer_cc
source venv_transformer_cc/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers scikit-learn pandas numpy matplotlib seaborn psutil tqdm
```

### 3. Generate Optimal Configuration
```bash
python config_cc.py
```

## 🎯 Usage

### Interactive Testing
```bash
# Activate environment
source venv_transformer_cc/bin/activate

# Test configuration
python config_cc.py

# Run on single GPU (for testing)
python src/transformer.py
```

### Production Training (SLURM)
```bash
# Edit submit_transformer_cc.sh with your account details
# Then submit:
sbatch submit_transformer_cc.sh
```

### Monitor Training
```bash
# Real-time monitoring
./monitor_training.sh

# Check logs
tail -f logs/training_*.log
tail -f logs/gpu_monitor.log
```

## 📊 Performance Optimizations

### Automatic Resource Scaling

| GPU Type | Memory | Batch Size | Latent Dim | Layers |
|----------|--------|------------|------------|--------|
| H100     | 80GB   | 128        | 768        | 8      |
| A100     | 40GB   | 96         | 512        | 6      |
| V100     | 32GB   | 64         | 384        | 4      |
| Others   | <32GB  | 32         | 256        | 3      |

### Cluster-Specific Optimizations

#### Cedar/Graham
- InfiniBand optimization: `mlx5_0,mlx5_2`
- Gradient accumulation: 2 steps
- Memory efficiency: High

#### Beluga  
- InfiniBand: `mlx5_0`
- GPU memory fraction: 85%
- Optimized for diverse workloads

#### Narval (Latest)
- HDR InfiniBand: `mlx5_0,mlx5_3,mlx5_6,mlx5_9`
- GPU memory fraction: 90%
- Advanced mixed precision

## 📁 Output Structure

```
results/
├── {log_type}/
│   ├── training_{log_type}.log          # Detailed training logs
│   ├── step_*.json                      # Step-by-step results
│   ├── metrics_{log_type}.json          # Training metrics
│   ├── results_{log_type}.pkl           # Predictions and analysis
│   └── visualization_{log_type}.png     # t-SNE plots

models/
├── transformer_{log_type}.pth           # Saved models
└── checkpoints/                         # Training checkpoints

logs/
├── distributed_training.log             # Distributed training logs
├── gpu_monitor.log                      # GPU utilization
└── memory_monitor.log                   # Memory usage
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Performance tuning
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Distributed training
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
```

### Custom Configuration
```python
# config_cc.py customization
config = ComputeCanadaConfig(
    batch_size_per_gpu=64,      # Adjust based on memory
    max_epochs=100,             # Training duration
    learning_rate=1e-3,         # Learning rate
    latent_dim=512,             # Model size
    max_memory_gb=200.0,        # Memory limit
    enable_distributed=True     # Multi-GPU training
)
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=32
   
   # Enable gradient checkpointing
   export GRADIENT_CHECKPOINTING=1
   ```

2. **NCCL Initialization Failed**
   ```bash
   # Check InfiniBand
   ibstat
   
   # Reset NCCL settings
   unset NCCL_IB_HCA
   export NCCL_IB_DISABLE=1
   ```

3. **Module Loading Issues**
   ```bash
   # Check available modules
   module avail cuda
   module avail python
   
   # Load compatible versions
   module load python/3.11 cuda/12.1
   ```

### Performance Issues
- **Slow Training**: Increase `num_workers` or reduce `chunk_size`
- **Memory Issues**: Enable `smart_subsampling` or reduce `max_samples_per_type`
- **GPU Underutilization**: Increase `batch_size_per_gpu` or enable `gradient_accumulation`

## 📈 Monitoring and Metrics

### Real-time Monitoring
```bash
# GPU utilization
watch nvidia-smi

# Training progress  
tail -f results/*/training_*.log

# System resources
htop
```

### Saved Metrics
- Training loss progression
- GPU memory usage over time
- Reconstruction quality metrics
- Multi-label prediction statistics
- System resource utilization

## 🔄 Multi-Node Training

For very large datasets or models:

```bash
# Multi-node SLURM script
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4

# Set distributed environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=16  # 4 nodes * 4 GPUs

srun python src/transformer.py
```

## 📚 Advanced Usage

### Custom Data Loading
```python
# Modify load_and_preprocess_data() in transformer.py
def load_custom_data(data_path):
    # Your custom loading logic
    return embeddings, labels
```

### Model Architecture Customization
```python
# Adjust in ComputeCanadaConfig
config.latent_dim = 1024           # Larger model
config.num_transformer_layers = 12 # Deeper network
config.num_attention_heads = 16    # More attention heads
```

### Hyperparameter Tuning
```python
# Grid search configuration
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
batch_sizes = [32, 64, 128]
latent_dims = [256, 512, 768]

# Use with Ray Tune or similar frameworks
```

## 📞 Support

- **Compute Canada Documentation**: [docs.computecanada.ca](https://docs.computecanada.ca)
- **PyTorch Distributed**: [pytorch.org/tutorials/distributed](https://pytorch.org/tutorials/distributed)
- **SLURM Guide**: [slurm.schedmd.com](https://slurm.schedmd.com)

## 🔄 Updates and Maintenance

The system automatically:
- Detects hardware changes
- Optimizes batch sizes
- Adjusts memory allocation
- Updates distributed settings
- Saves configurations for reproducibility

## 📄 License

This implementation is optimized for academic and research use on Compute Canada clusters.

---

**Pro Tips for Compute Canada Users:**

1. **Resource Allocation**: Always specify exact GPU types in SLURM (`--constraint=cascade`)
2. **Data Location**: Use `$SCRATCH` for large datasets, `$PROJECT` for long-term storage
3. **Monitoring**: Set up email notifications for long jobs
4. **Efficiency**: Use `--exclusive` flag for dedicated node access
5. **Debugging**: Start with small jobs to validate configuration

Happy training! 🚀 