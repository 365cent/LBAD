#!/usr/bin/env python3
"""
Compute Canada Setup Script
Automatically configures the environment for optimal transformer training
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

def run_command(cmd, check=True, capture_output=False):
    """Run shell command with error handling"""
    print(f"Running: {cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e}")
        return False

def check_requirements():
    """Check system requirements and available resources"""
    print("=" * 60)
    print("CHECKING SYSTEM REQUIREMENTS")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or python_version.minor < 8:
        print("WARNING: Python 3.8+ recommended")
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            n_gpus = torch.cuda.device_count()
            print(f"Number of GPUs: {n_gpus}")
            for i in range(n_gpus):
                gpu_name = torch.cuda.get_device_properties(i).name
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    except ImportError:
        print("PyTorch not installed - will be installed during setup")
    
    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Total Memory: {memory_gb:.1f} GB")
        cpu_count = psutil.cpu_count()
        print(f"CPU Cores: {cpu_count}")
    except ImportError:
        print("psutil not available - will be installed")
    
    # Check available modules on cluster
    print("\nChecking available modules...")
    modules_output = run_command("module avail 2>&1 | grep -E '(python|cuda|torch)'", 
                                capture_output=True, check=False)
    if modules_output:
        print("Available relevant modules:")
        print(modules_output)
    
    print("=" * 60)

def setup_environment():
    """Setup Python environment and dependencies"""
    print("SETTING UP ENVIRONMENT")
    print("=" * 60)
    
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv_transformer_cc")
    if not venv_path.exists():
        print("Creating virtual environment...")
        if not run_command(f"python -m venv {venv_path}"):
            print("Failed to create virtual environment")
            return False
    
    # Activate virtual environment (for the script)
    activate_script = venv_path / "bin" / "activate"
    if activate_script.exists():
        print(f"Virtual environment created at: {venv_path}")
    
    return True

def install_dependencies(use_cuda=True):
    """Install required Python packages"""
    print("INSTALLING DEPENDENCIES")
    print("=" * 60)
    
    # Base packages
    packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "psutil>=5.8.0",
        "tqdm>=4.60.0"
    ]
    
    # PyTorch installation
    if use_cuda:
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    print("Installing PyTorch...")
    if not run_command(torch_cmd):
        print("Failed to install PyTorch")
        return False
    
    # Install other packages
    for package in packages:
        print(f"Installing {package}...")
        if not run_command(f"pip install {package}"):
            print(f"Failed to install {package}")
            return False
    
    print("All dependencies installed successfully!")
    return True

def setup_directories():
    """Create necessary directories"""
    print("SETTING UP DIRECTORIES")
    print("=" * 60)
    
    directories = [
        "results",
        "models", 
        "logs",
        "configs",
        "checkpoints"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return True

def create_launcher_script():
    """Create optimized launcher script"""
    print("CREATING LAUNCHER SCRIPT")
    print("=" * 60)
    
    launcher_content = """#!/bin/bash
# Optimized launcher for Compute Canada

# Source environment
source venv_transformer_cc/bin/activate

# Load optimal configuration
python config_cc.py

# Run transformer training with optimal settings
echo "Starting transformer training..."
echo "Configuration:"
echo "  - Auto-detected resources"
echo "  - Multi-GPU support enabled"
echo "  - Mixed precision training"
echo "  - Optimized memory management"

# Set environment variables
source setup_cc_env.sh

# Run main script
python src/transformer.py "$@"

echo "Training completed!"
"""
    
    with open("launch_transformer.sh", "w") as f:
        f.write(launcher_content)
    
    os.chmod("launch_transformer.sh", 0o755)
    print("Created launcher script: launch_transformer.sh")
    
    return True

def create_monitoring_script():
    """Create monitoring script for long-running jobs"""
    monitoring_content = """#!/bin/bash
# Job monitoring script for Compute Canada

echo "=== Transformer Training Monitor ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Monitor GPU usage
monitor_gpus() {
    while true; do
        echo "$(date): GPU Status"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
        echo "---"
        sleep 300  # Every 5 minutes
    done > logs/gpu_monitor.log &
}

# Monitor memory usage
monitor_memory() {
    while true; do
        echo "$(date): Memory Usage"
        free -h
        echo "---"
        sleep 300
    done > logs/memory_monitor.log &
}

# Start monitoring if GPUs available
if command -v nvidia-smi &> /dev/null; then
    monitor_gpus
    echo "GPU monitoring started"
fi

monitor_memory
echo "Memory monitoring started"

# Wait for main process
wait

echo "Monitoring completed at $(date)"
"""
    
    with open("monitor_training.sh", "w") as f:
        f.write(monitoring_content)
    
    os.chmod("monitor_training.sh", 0o755)
    print("Created monitoring script: monitor_training.sh")

def update_slurm_script(account=None, email=None):
    """Update SLURM script with user details"""
    if not account or not email:
        return
    
    script_path = Path("submit_transformer_cc.sh")
    if script_path.exists():
        content = script_path.read_text()
        content = content.replace("def-youraccountname", f"def-{account}")
        content = content.replace("your.email@example.com", email)
        script_path.write_text(content)
        print(f"Updated SLURM script with account: {account}")

def main():
    parser = argparse.ArgumentParser(description="Setup Compute Canada environment for transformer training")
    parser.add_argument("--account", type=str, help="Compute Canada account name")
    parser.add_argument("--email", type=str, help="Email for job notifications")
    parser.add_argument("--no-cuda", action="store_true", help="Install CPU-only PyTorch")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()
    
    print("🚀 COMPUTE CANADA TRANSFORMER SETUP")
    print("=" * 60)
    
    # Check requirements
    check_requirements()
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return 1
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies(use_cuda=not args.no_cuda):
            print("❌ Dependency installation failed")
            return 1
    
    # Setup directories
    if not setup_directories():
        print("❌ Directory setup failed")
        return 1
    
    # Create scripts
    if not create_launcher_script():
        print("❌ Launcher script creation failed")
        return 1
    
    create_monitoring_script()
    
    # Update SLURM script
    if args.account and args.email:
        update_slurm_script(args.account, args.email)
    
    print("\n✅ SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Next steps:")
    print("1. Activate environment: source venv_transformer_cc/bin/activate")
    print("2. Test configuration: python config_cc.py")
    print("3. Submit job: sbatch submit_transformer_cc.sh")
    print("4. Monitor progress: ./monitor_training.sh")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 