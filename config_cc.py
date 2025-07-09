#!/usr/bin/env python3
"""
Compute Canada Supercomputing Configuration
Auto-detects resources and optimizes settings for various cluster configurations
"""

import os
import torch
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ComputeCanadaConfig:
    """Configuration optimized for Compute Canada clusters"""
    
    # Resource Detection
    auto_detect_resources: bool = True
    
    # GPU Configuration
    use_mixed_precision: bool = True
    max_gpu_memory_fraction: float = 0.9
    enable_gradient_checkpointing: bool = True
    
    # Training Configuration
    batch_size_per_gpu: int = 64  # Will be auto-adjusted
    gradient_accumulation_steps: int = 4
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    
    # Memory Management
    max_memory_gb: float = 100.0  # Will be auto-detected
    chunk_size: int = 10000
    pin_memory: bool = True
    num_workers: int = 8  # Will be auto-adjusted
    
    # Model Architecture
    latent_dim: int = 512
    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # Data Processing
    max_samples_per_type: int = 1000000
    enable_smart_subsampling: bool = True
    normalize_embeddings: bool = True
    
    # I/O Configuration
    save_checkpoints: bool = True
    checkpoint_every_n_epochs: int = 10
    save_predictions: bool = True
    create_visualizations: bool = True
    log_level: str = "INFO"
    
    # Cluster-specific optimizations
    cluster_type: str = "auto"  # auto, cedar, graham, beluga, narval
    enable_distributed: bool = True
    backend: str = "nccl"
    
    def __post_init__(self):
        """Auto-configure based on detected resources"""
        if self.auto_detect_resources:
            self._detect_and_configure()
    
    def _detect_and_configure(self):
        """Detect system resources and optimize configuration"""
        # Detect cluster type
        hostname = os.uname().nodename
        if 'cedar' in hostname:
            self.cluster_type = 'cedar'
            self._configure_cedar()
        elif 'graham' in hostname:
            self.cluster_type = 'graham'
            self._configure_graham()
        elif 'beluga' in hostname:
            self.cluster_type = 'beluga'
            self._configure_beluga()
        elif 'narval' in hostname:
            self.cluster_type = 'narval'
            self._configure_narval()
        else:
            self.cluster_type = 'generic'
            self._configure_generic()
        
        # GPU configuration
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Adjust batch size based on GPU memory
            if gpu_memory >= 80:  # H100
                self.batch_size_per_gpu = 128
                self.latent_dim = 768
                self.num_transformer_layers = 8
            elif gpu_memory >= 40:  # A100
                self.batch_size_per_gpu = 96
                self.latent_dim = 512
                self.num_transformer_layers = 6
            elif gpu_memory >= 24:  # RTX 6000, A6000
                self.batch_size_per_gpu = 64
                self.latent_dim = 384
                self.num_transformer_layers = 4
            else:  # Smaller GPUs
                self.batch_size_per_gpu = 32
                self.latent_dim = 256
                self.num_transformer_layers = 3
        
        # Memory configuration
        total_memory = psutil.virtual_memory().total / (1024**3)
        self.max_memory_gb = min(total_memory * 0.8, 500.0)
        
        # CPU configuration
        n_cpus = psutil.cpu_count()
        self.num_workers = min(n_cpus // 2, 16)
        
        # Adjust parameters for very large systems
        if total_memory > 500:  # Large memory systems
            self.max_samples_per_type = 2000000
            self.chunk_size = 50000
        elif total_memory > 200:  # Medium systems
            self.max_samples_per_type = 1500000
            self.chunk_size = 25000
        else:  # Smaller systems
            self.max_samples_per_type = 1000000
            self.chunk_size = 10000
    
    def _configure_cedar(self):
        """Cedar-specific optimizations"""
        self.enable_gradient_checkpointing = True
        self.pin_memory = True
        # Cedar has fast interconnect
        self.enable_distributed = True
    
    def _configure_graham(self):
        """Graham-specific optimizations"""
        self.enable_gradient_checkpointing = True
        self.pin_memory = True
        # Graham optimizations
        self.gradient_accumulation_steps = 2
    
    def _configure_beluga(self):
        """Beluga-specific optimizations"""
        self.enable_gradient_checkpointing = True
        self.pin_memory = True
        # Beluga optimizations
        self.max_gpu_memory_fraction = 0.85
    
    def _configure_narval(self):
        """Narval-specific optimizations (newest cluster)"""
        self.enable_gradient_checkpointing = True
        self.pin_memory = True
        # Narval has latest hardware
        self.use_mixed_precision = True
        self.max_gpu_memory_fraction = 0.9
    
    def _configure_generic(self):
        """Generic configuration for unknown systems"""
        self.enable_gradient_checkpointing = True
        self.pin_memory = True
        self.max_gpu_memory_fraction = 0.8
    
    def get_effective_batch_size(self, n_gpus: int = 1) -> int:
        """Calculate effective batch size accounting for gradient accumulation"""
        return self.batch_size_per_gpu * n_gpus * self.gradient_accumulation_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def save(self, path: Path):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'ComputeCanadaConfig':
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("COMPUTE CANADA CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Cluster Type: {self.cluster_type}")
        print(f"GPUs Available: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Type: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        print(f"Total System Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"CPU Cores: {psutil.cpu_count()}")
        print()
        print("Training Configuration:")
        print(f"  Batch Size per GPU: {self.batch_size_per_gpu}")
        print(f"  Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {self.get_effective_batch_size()}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Max Epochs: {self.max_epochs}")
        print(f"  Mixed Precision: {self.use_mixed_precision}")
        print()
        print("Model Configuration:")
        print(f"  Latent Dimension: {self.latent_dim}")
        print(f"  Transformer Layers: {self.num_transformer_layers}")
        print(f"  Attention Heads: {self.num_attention_heads}")
        print(f"  Dropout Rate: {self.dropout_rate}")
        print()
        print("Memory Configuration:")
        print(f"  Max Memory: {self.max_memory_gb:.1f} GB")
        print(f"  Chunk Size: {self.chunk_size:,}")
        print(f"  Max Samples: {self.max_samples_per_type:,}")
        print(f"  Num Workers: {self.num_workers}")
        print("=" * 60)

def get_optimal_config() -> ComputeCanadaConfig:
    """Get optimally configured settings for current environment"""
    return ComputeCanadaConfig()

def create_environment_script(config: ComputeCanadaConfig, output_path: Path):
    """Create environment setup script for the current configuration"""
    script_content = f"""#!/bin/bash
# Auto-generated environment setup for Compute Canada
# Generated for cluster type: {config.cluster_type}

# Set optimal environment variables
export OMP_NUM_THREADS={config.num_workers}
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Memory optimization
export MALLOC_TRIM_THRESHOLD=100000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Distributed training
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Cluster-specific optimizations
"""
    
    if config.cluster_type in ['cedar', 'graham']:
        script_content += """
# Cedar/Graham specific
export NCCL_IB_HCA=mlx5_0,mlx5_2
"""
    elif config.cluster_type == 'beluga':
        script_content += """
# Beluga specific
export NCCL_IB_HCA=mlx5_0
"""
    elif config.cluster_type == 'narval':
        script_content += """
# Narval specific (InfiniBand HDR)
export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_6,mlx5_9
export NCCL_NET_GDR_LEVEL=3
"""
    
    script_content += f"""
echo "Environment configured for {config.cluster_type}"
echo "GPU Memory Fraction: {config.max_gpu_memory_fraction}"
echo "Batch Size per GPU: {config.batch_size_per_gpu}"
echo "Workers: {config.num_workers}"
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_path, 0o755)
    print(f"Environment script created: {output_path}")

if __name__ == "__main__":
    # Create and display optimal configuration
    config = get_optimal_config()
    config.print_summary()
    
    # Save configuration
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config.save(config_dir / "cc_optimal_config.json")
    
    # Create environment script
    create_environment_script(config, Path("setup_cc_env.sh"))
    
    print(f"\nConfiguration files created in: {config_dir}")
    print("Environment setup script: setup_cc_env.sh") 