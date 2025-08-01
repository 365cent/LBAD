#!/usr/bin/env python3
"""
Systematic debugging to find the exact hanging point
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys

def log_with_timestamp(msg):
    """Log message with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def test_basic_operations():
    """Test basic PyTorch operations"""
    log_with_timestamp("🧪 Testing basic PyTorch operations...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_with_timestamp(f"Using device: {device}")
    
    # Test 1: Basic tensor operations
    log_with_timestamp("Test 1: Creating tensors...")
    x = torch.randn(10, 2314).to(device)
    y = torch.randn(10, 4).to(device)
    log_with_timestamp("✅ Tensors created successfully")
    
    # Test 2: Simple linear layer
    log_with_timestamp("Test 2: Testing linear layer...")
    linear = nn.Linear(2314, 256).to(device)
    z = linear(x)
    log_with_timestamp(f"✅ Linear layer output shape: {z.shape}")
    
    # Test 3: Loss computation
    log_with_timestamp("Test 3: Testing loss computation...")
    pred = torch.randn(10, 4).to(device)
    loss = F.binary_cross_entropy_with_logits(pred, y)
    log_with_timestamp(f"✅ Loss computed: {loss.item():.4f}")
    
    return True

def test_simple_model():
    """Test our simplified model"""
    log_with_timestamp("🧪 Testing simplified model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple model definition
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(2314, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, 256),
            )
            self.decoder = nn.Linear(256, 2314)
            self.classifier = nn.Linear(256, 4)
        
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            labels = self.classifier(z)
            return {"reconstructed": recon, "labels": labels}
    
    log_with_timestamp("Creating model...")
    model = TestModel().to(device)
    log_with_timestamp(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    log_with_timestamp("Testing forward pass...")
    x = torch.randn(10, 2314).to(device)
    
    log_with_timestamp("Calling model.forward()...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(x)
    
    forward_time = time.time() - start_time
    log_with_timestamp(f"✅ Forward pass completed in {forward_time:.3f}s")
    log_with_timestamp(f"   Reconstructed shape: {outputs['reconstructed'].shape}")
    log_with_timestamp(f"   Labels shape: {outputs['labels'].shape}")
    
    return True

def test_training_step():
    """Test a single training step"""
    log_with_timestamp("🧪 Testing single training step...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(2314, 256),
        nn.ReLU(),
        nn.Linear(256, 2314)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test data
    x = torch.randn(10, 2314).to(device)
    
    log_with_timestamp("Starting training step...")
    
    log_with_timestamp("1. optimizer.zero_grad()")
    optimizer.zero_grad()
    
    log_with_timestamp("2. Forward pass")
    pred = model(x)
    
    log_with_timestamp("3. Loss computation")
    loss = F.mse_loss(pred, x)
    log_with_timestamp(f"   Loss: {loss.item():.4f}")
    
    log_with_timestamp("4. Backward pass")
    loss.backward()
    
    log_with_timestamp("5. Gradient clipping")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    log_with_timestamp("6. Optimizer step")
    optimizer.step()
    
    log_with_timestamp("✅ Training step completed successfully")
    
    return True

def test_dataloader():
    """Test DataLoader functionality"""
    log_with_timestamp("🧪 Testing DataLoader...")
    
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create simple dataset
    x = torch.randn(10, 2314)
    y = torch.randn(10, 4)
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)
    
    log_with_timestamp(f"DataLoader created with {len(dataloader)} batches")
    
    log_with_timestamp("Iterating through batches...")
    for i, (batch_x, batch_y) in enumerate(dataloader):
        log_with_timestamp(f"  Batch {i+1}: x={batch_x.shape}, y={batch_y.shape}")
    
    log_with_timestamp("✅ DataLoader iteration completed")
    
    return True

def main():
    """Run systematic debugging tests"""
    log_with_timestamp("🚀 Starting systematic debugging...")
    
    tests = [
        ("Basic Operations", test_basic_operations),
        ("Simple Model", test_simple_model), 
        ("Training Step", test_training_step),
        ("DataLoader", test_dataloader),
    ]
    
    for test_name, test_func in tests:
        log_with_timestamp(f"\n{'='*50}")
        log_with_timestamp(f"Running: {test_name}")
        log_with_timestamp('='*50)
        
        try:
            start_time = time.time()
            success = test_func()
            test_time = time.time() - start_time
            
            if success:
                log_with_timestamp(f"✅ {test_name} PASSED in {test_time:.3f}s")
            else:
                log_with_timestamp(f"❌ {test_name} FAILED")
                break
                
        except Exception as e:
            log_with_timestamp(f"💥 {test_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            break
        
        log_with_timestamp("")
    
    log_with_timestamp("🏁 Debugging session completed")

if __name__ == "__main__":
    main()