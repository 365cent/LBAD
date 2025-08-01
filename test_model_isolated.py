#!/usr/bin/env python3
"""
Test the exact model architecture in complete isolation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def test_isolated_model():
    """Test the exact same model as in transformer.py"""
    
    print("🧪 Testing isolated model...")
    
    # Exact same model definition
    class TestModel(nn.Module):
        def __init__(self, input_dim=2314, latent_dim=256, n_labels=4):
            super().__init__()
            self.encoder = nn.Linear(input_dim, latent_dim)
            self.decoder = nn.Linear(latent_dim, input_dim)
            self.classifier = nn.Linear(latent_dim, n_labels)
        
        def forward(self, x, **kwargs):
            z = F.relu(self.encoder(x))
            reconstructed = self.decoder(z)
            labels = self.classifier(z)
            return {
                "reconstructed": reconstructed,
                "labels": labels,
            }
    
    # Test on CPU first
    print("📱 Testing on CPU...")
    model_cpu = TestModel()
    x_cpu = torch.randn(10, 2314)
    
    start_time = time.time()
    outputs_cpu = model_cpu(x_cpu)
    cpu_time = time.time() - start_time
    print(f"✅ CPU forward pass: {cpu_time:.3f}s")
    print(f"   Shapes: recon={outputs_cpu['reconstructed'].shape}, labels={outputs_cpu['labels'].shape}")
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        print("🚀 Testing on CUDA...")
        model_cuda = TestModel().cuda()
        x_cuda = torch.randn(10, 2314).cuda()
        
        print("   Calling forward pass...")
        start_time = time.time()
        
        # Add timeout
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("CUDA forward timed out!")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        try:
            outputs_cuda = model_cuda(x_cuda)
            signal.alarm(0)
            cuda_time = time.time() - start_time
            print(f"✅ CUDA forward pass: {cuda_time:.3f}s")
            print(f"   Shapes: recon={outputs_cuda['reconstructed'].shape}, labels={outputs_cuda['labels'].shape}")
        except TimeoutError:
            signal.alarm(0)
            print("❌ CUDA forward pass TIMED OUT - confirms CUDA issue!")
            return False
    
    return True

if __name__ == "__main__":
    success = test_isolated_model()
    print(f"\n🎯 Test result: {'PASSED' if success else 'FAILED'}")