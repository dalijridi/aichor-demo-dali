#!/usr/bin/env python3
import time
import sys
import subprocess
import os

def log_with_flush(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()
    sys.stderr.flush()

log_with_flush("🚀 GPU RUNTIME DETECTION TEST")
log_with_flush("=" * 50)

# Check environment
log_with_flush("🔍 CHECKING ENVIRONMENT:")
log_with_flush(f"   NVIDIA_VISIBLE_DEVICES: {os.getenv('NVIDIA_VISIBLE_DEVICES', 'not set')}")

# Check for GPU devices
log_with_flush("\n🔍 CHECKING GPU DEVICES:")
try:
    result = subprocess.run(['ls', '/dev/'], capture_output=True, text=True)
    gpu_devices = [line for line in result.stdout.split('\n') if 'nvidia' in line.lower()]
    if gpu_devices:
        log_with_flush(f"   ✅ Found GPU devices: {gpu_devices}")
    else:
        log_with_flush("   ❌ No GPU devices found in /dev/")
except Exception as e:
    log_with_flush(f"   ❌ Error checking devices: {e}")

# Check nvidia-smi
log_with_flush("\n🔍 CHECKING NVIDIA-SMI:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        log_with_flush("   ✅ nvidia-smi successful!")
        lines = result.stdout.split('\n')[:10]  # First 10 lines
        for line in lines:
            if line.strip():
                log_with_flush(f"   {line}")
    else:
        log_with_flush(f"   ❌ nvidia-smi failed: {result.stderr}")
except Exception as e:
    log_with_flush(f"   ❌ nvidia-smi error: {e}")

# Check PyTorch
log_with_flush("\n🔍 CHECKING PYTORCH:")
try:
    import torch
    log_with_flush(f"   ✅ PyTorch version: {torch.__version__}")
    log_with_flush(f"   🎯 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        log_with_flush(f"   🎯 CUDA version: {torch.version.cuda}")
        log_with_flush(f"   🎯 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            log_with_flush(f"   🎯 GPU {i}: {name} ({memory:.1f} GB)")
            
        # Test GPU computation
        log_with_flush("\n🧮 TESTING GPU COMPUTATION:")
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        start_time = time.time()
        z = torch.matmul(x, y)
        gpu_time = time.time() - start_time
        
        log_with_flush(f"   ✅ GPU matrix multiplication successful!")
        log_with_flush(f"   ⏱️  GPU compute time: {gpu_time*1000:.2f}ms")
        
    else:
        log_with_flush("   ⚠️  CUDA not available - running on CPU")
        
        # Test CPU computation for comparison
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        
        start_time = time.time()
        z = torch.matmul(x, y)
        cpu_time = time.time() - start_time
        
        log_with_flush(f"   ✅ CPU matrix multiplication successful!")
        log_with_flush(f"   ⏱️  CPU compute time: {cpu_time*1000:.2f}ms")
        
except Exception as e:
    log_with_flush(f"   ❌ PyTorch error: {e}")

log_with_flush("\n" + "=" * 50)
log_with_flush("🎉 GPU DETECTION TEST COMPLETED!")
log_with_flush("=" * 50)
