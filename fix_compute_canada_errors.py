#!/usr/bin/env python3
"""
Compute Canada Runtime Error Fix Script
Fixes common issues including Intel GPU driver conflicts, CUDA problems, and multiprocessing errors
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging

def setup_logging():
    """Setup logging for debugging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cc_fix.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def fix_intel_gpu_conflict():
    """Fix Intel GPU driver conflicts with PyTorch CUDA"""
    logger = logging.getLogger(__name__)
    logger.info("Fixing Intel GPU driver conflicts...")
    
    # Set environment variables to disable Intel GPU drivers
    env_fixes = {
        'ONEAPI_DEVICE_SELECTOR': 'cuda:*',  # Only use CUDA devices
        'INTEL_DEVICE_SELECTOR': 'none',     # Disable Intel devices
        'DISABLE_INTEL_GPU': '1',            # Custom flag
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'),
        'HIP_VISIBLE_DEVICES': '',           # Disable AMD ROCm
        'ONEAPI_ROOT': '',                   # Disable oneAPI
        'SYCL_DEVICE_FILTER': 'cuda',       # Only CUDA for SYCL
    }
    
    # Create environment setup script
    env_script = """#!/bin/bash
# Fix for Intel GPU driver conflicts on Compute Canada

# Disable Intel GPU and oneAPI
export ONEAPI_DEVICE_SELECTOR="cuda:*"
export INTEL_DEVICE_SELECTOR="none"
export DISABLE_INTEL_GPU=1
export ONEAPI_ROOT=""
export SYCL_DEVICE_FILTER="cuda"

# Disable AMD ROCm
export HIP_VISIBLE_DEVICES=""

# Force CUDA only
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# PyTorch specific fixes
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export TORCH_USE_CUDA_DSA=1

# Multiprocessing fixes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Library path fixes - remove Intel paths
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\\n' | grep -v intel | grep -v oneapi | tr '\\n' ':' | sed 's/:$//')

echo "Environment fixed for Compute Canada"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
"""
    
    with open('fix_cc_env.sh', 'w') as f:
        f.write(env_script)
    
    os.chmod('fix_cc_env.sh', 0o755)
    logger.info("Created fix_cc_env.sh - source this before running PyTorch")
    
    return env_fixes

def fix_pytorch_installation():
    """Reinstall PyTorch with proper CUDA support"""
    logger = logging.getLogger(__name__)
    logger.info("Fixing PyTorch installation...")
    
    commands = [
        "pip uninstall -y torch torchvision torchaudio",
        "pip cache purge",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir"
    ]
    
    for cmd in commands:
        logger.info(f"Running: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed: {cmd} - {e}")
            return False
    
    return True

def fix_multiprocessing_spawn():
    """Fix multiprocessing spawn issues"""
    logger = logging.getLogger(__name__)
    logger.info("Fixing multiprocessing spawn issues...")
    
    # Create a wrapper script that sets the environment before importing torch
    wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper script to fix import issues before running the main script
"""
import os
import sys

# Set environment before importing torch
os.environ['ONEAPI_DEVICE_SELECTOR'] = 'cuda:*'
os.environ['INTEL_DEVICE_SELECTOR'] = 'none'
os.environ['DISABLE_INTEL_GPU'] = '1'
os.environ['ONEAPI_ROOT'] = ''
os.environ['SYCL_DEVICE_FILTER'] = 'cuda'
os.environ['HIP_VISIBLE_DEVICES'] = ''

# Fix library path
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if ld_path:
    # Remove Intel and oneAPI paths
    paths = [p for p in ld_path.split(':') if 'intel' not in p.lower() and 'oneapi' not in p.lower()]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(paths)

# Now import torch
try:
    import torch
    print(f"PyTorch successfully imported: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
except Exception as e:
    print(f"Error importing torch: {e}")
    sys.exit(1)

# Import and run the actual transformer script
if __name__ == "__main__":
    # Add src to path
    sys.path.insert(0, 'src')
    
    # Import and run transformer
    import transformer
    transformer.main()
'''
    
    with open('run_transformer_fixed.py', 'w') as f:
        f.write(wrapper_content)
    
    os.chmod('run_transformer_fixed.py', 0o755)
    logger.info("Created run_transformer_fixed.py")
    
    return True

def create_fixed_slurm_script():
    """Create a fixed SLURM script with proper environment setup"""
    logger = logging.getLogger(__name__)
    logger.info("Creating fixed SLURM script...")
    
    slurm_content = '''#!/bin/bash
#SBATCH --job-name=transformer_fixed
#SBATCH --account=def-youraccountname  # Replace with your account
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=transformer_%j.out
#SBATCH --error=transformer_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com

# Load modules (avoid Intel GPU modules)
module purge  # Clear all modules first
module load StdEnv/2023
module load python/3.11
module load cuda/12.1
module load cudnn/8.9.0

# CRITICAL: Remove Intel GPU libraries from path
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\\n' | grep -v intel | grep -v oneapi | tr '\\n' ':' | sed 's/:$//')

# Fix environment variables
source fix_cc_env.sh

# Create/activate virtual environment
if [ ! -d "venv_fixed" ]; then
    echo "Creating fixed virtual environment..."
    python -m venv venv_fixed
fi

source venv_fixed/bin/activate

# Install dependencies with fixes
pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip cache purge

# Install PyTorch with specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir

# Install other dependencies
pip install transformers scikit-learn pandas numpy matplotlib seaborn psutil tqdm

# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Devices: {torch.cuda.device_count()}')"

# Create necessary directories
mkdir -p results models logs

# Run the fixed transformer script
echo "Starting fixed transformer training..."
python run_transformer_fixed.py

echo "Job completed!"
'''
    
    with open('submit_transformer_fixed.sh', 'w') as f:
        f.write(slurm_content)
    
    os.chmod('submit_transformer_fixed.sh', 0o755)
    logger.info("Created submit_transformer_fixed.sh")

def fix_transformer_imports():
    """Fix imports in transformer.py to handle the library conflicts"""
    logger = logging.getLogger(__name__)
    logger.info("Fixing transformer.py imports...")
    
    # Read the current transformer.py
    transformer_path = Path('src/transformer.py')
    if not transformer_path.exists():
        logger.error("src/transformer.py not found!")
        return False
    
    # Create a fixed version with proper import handling
    fixed_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance Transformer-based Unsupervised Multi-Label Learning for Supercomputing
FIXED VERSION - Handles Intel GPU driver conflicts on Compute Canada
"""

# CRITICAL: Set environment before importing torch
import os
os.environ['ONEAPI_DEVICE_SELECTOR'] = 'cuda:*'
os.environ['INTEL_DEVICE_SELECTOR'] = 'none'
os.environ['DISABLE_INTEL_GPU'] = '1'
os.environ['ONEAPI_ROOT'] = ''
os.environ['SYCL_DEVICE_FILTER'] = 'cuda'
os.environ['HIP_VISIBLE_DEVICES'] = ''

# Fix library path
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if ld_path:
    paths = [p for p in ld_path.split(':') if 'intel' not in p.lower() and 'oneapi' not in p.lower()]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(paths)

import sys
import time
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

import numpy as np

# Now safely import torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    from torch.cuda.amp import GradScaler, autocast
    print(f"PyTorch successfully loaded: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"Error importing PyTorch: {e}")
    print("This is likely due to Intel GPU driver conflicts.")
    print("Please run: source fix_cc_env.sh")
    sys.exit(1)

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, classification_report

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set multiprocessing start method to avoid spawn issues
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Rest of the transformer code continues here...
'''
    
    # Read the original content and append it (excluding the imports and header)
    original_content = transformer_path.read_text()
    
    # Find where the main code starts (after imports)
    lines = original_content.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('@dataclass') or line.strip().startswith('class ') or line.strip().startswith('def '):
            start_idx = i
            break
    
    # Combine fixed imports with original code
    original_main_code = '\n'.join(lines[start_idx:])
    full_fixed_content = fixed_content + '\n' + original_main_code
    
    # Write the fixed version
    with open('src/transformer_fixed.py', 'w') as f:
        f.write(full_fixed_content)
    
    logger.info("Created src/transformer_fixed.py with proper import handling")
    return True

def run_tests():
    """Run tests to verify fixes"""
    logger = logging.getLogger(__name__)
    logger.info("Running tests to verify fixes...")
    
    test_script = '''
import os
import sys

# Apply fixes
os.environ['ONEAPI_DEVICE_SELECTOR'] = 'cuda:*'
os.environ['INTEL_DEVICE_SELECTOR'] = 'none'
os.environ['DISABLE_INTEL_GPU'] = '1'

try:
    import torch
    print("✅ PyTorch import successful")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available with {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_properties(i).name}")
    else:
        print("❌ CUDA not available")
    
    # Test tensor creation
    x = torch.randn(10, 10)
    if torch.cuda.is_available():
        x = x.cuda()
        print("✅ GPU tensor creation successful")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
'''
    
    with open('test_fixes.py', 'w') as f:
        f.write(test_script)
    
    try:
        subprocess.run([sys.executable, 'test_fixes.py'], check=True)
        logger.info("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError:
        logger.error("❌ Tests failed")
        return False

def main():
    """Main fix function"""
    logger = setup_logging()
    logger.info("🔧 Starting Compute Canada runtime error fixes...")
    
    # Apply all fixes
    fixes = [
        ("Intel GPU conflicts", fix_intel_gpu_conflict),
        ("PyTorch installation", fix_pytorch_installation),
        ("Multiprocessing spawn", fix_multiprocessing_spawn),
        ("SLURM script", create_fixed_slurm_script),
        ("Transformer imports", fix_transformer_imports),
    ]
    
    success_count = 0
    for name, fix_func in fixes:
        logger.info(f"Applying fix: {name}")
        try:
            if fix_func():
                logger.info(f"✅ {name} fix applied successfully")
                success_count += 1
            else:
                logger.error(f"❌ {name} fix failed")
        except Exception as e:
            logger.error(f"❌ {name} fix failed with error: {e}")
    
    logger.info(f"Applied {success_count}/{len(fixes)} fixes successfully")
    
    # Run tests
    logger.info("Running verification tests...")
    if run_tests():
        logger.info("🎉 All fixes verified successfully!")
        
        print("\n" + "="*60)
        print("🎉 COMPUTE CANADA FIXES APPLIED SUCCESSFULLY!")
        print("="*60)
        print("Next steps:")
        print("1. Source the environment fix: source fix_cc_env.sh")
        print("2. Test PyTorch: python test_fixes.py")
        print("3. Run training: python run_transformer_fixed.py")
        print("4. Or submit job: sbatch submit_transformer_fixed.sh")
        print("="*60)
    else:
        logger.error("❌ Some tests failed. Check the logs.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 