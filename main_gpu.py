#!/usr/bin/env python3
import sys
import time
import os
import logging
import subprocess

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Configure logging to write to multiple streams
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)

def force_log(message):
    """Multi-channel logging to ensure visibility"""
    # Method 1: Print to stdout/stderr
    print(f"[STDOUT] {message}")
    print(f"[STDERR] {message}", file=sys.stderr)
    
    # Method 2: Python logging
    logger.info(message)
    
    # Method 3: Write to file (for debugging)
    with open('/tmp/training.log', 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    # Force flush all streams
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Method 4: Echo to system (visible in container logs)
    try:
        subprocess.run(['echo', f"[ECHO] {message}"], check=False)
    except:
        pass

def main():
    force_log("🚀 STARTING VERTEX AI LOGGING TEST")
    force_log("=" * 60)
    
    # Environment info
    force_log("📋 ENVIRONMENT INFO:")
    force_log(f"   Python: {sys.version}")
    force_log(f"   Working dir: {os.getcwd()}")
    force_log(f"   User: {os.getenv('USER', 'unknown')}")
    
    # Test PyTorch
    force_log("\n🧠 TESTING PYTORCH:")
    try:
        import torch
        force_log(f"   ✅ PyTorch version: {torch.__version__}")
        force_log(f"   🔍 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            force_log(f"   🎯 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                force_log(f"   🎯 GPU {i}: {gpu_name}")
        else:
            force_log("   ⚠️  Running on CPU")
    except Exception as e:
        force_log(f"   ❌ PyTorch error: {e}")
    
    # Simple computation test
    force_log("\n🧮 TESTING COMPUTATION:")
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        force_log(f"   🎯 Using device: {device}")
        
        # Simple tensor operations
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        
        start_time = time.time()
        z = torch.matmul(x, y)
        compute_time = time.time() - start_time
        
        force_log(f"   ✅ Matrix multiplication completed")
        force_log(f"   ⏱️  Compute time: {compute_time*1000:.2f}ms")
        force_log(f"   📊 Result shape: {z.shape}")
        force_log(f"   📊 Result mean: {z.mean().item():.6f}")
        
    except Exception as e:
        force_log(f"   ❌ Computation error: {e}")
    
    # Progress simulation
    force_log("\n🏃 SIMULATING TRAINING PROGRESS:")
    for i in range(10):
        progress = (i + 1) / 10 * 100
        force_log(f"   Step {i+1}/10 - Progress: {progress:5.1f}%")
        time.sleep(1)
    
    # Final status
    force_log("\n" + "=" * 60)
    force_log("🎉 VERTEX AI LOGGING TEST COMPLETED SUCCESSFULLY!")
    force_log("🎉 If you see this message, logging is working!")
    force_log("=" * 60)
    
    # Show log file content
    try:
        with open('/tmp/training.log', 'r') as f:
            content = f.read()
        force_log(f"\n📄 LOG FILE CONTENT ({len(content)} chars):")
        force_log(content[:500] + "..." if len(content) > 500 else content)
    except:
        force_log("📄 Could not read log file")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        force_log(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
