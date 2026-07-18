import time
import torch
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

# 1. Load PyTorch model on CPU
print("Loading PyTorch model...")
pt_model = YOLO("best _bigcola.pt")

# 2. Load ONNX Session on CPU
print("Loading ONNX model...")
onnx_path = "best _bigcola.onnx"
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Create dummy input simulating a standard 640x640 frame
dummy_numpy = np.random.rand(1, 3, 640, 640).astype(np.float32)
dummy_tensor = torch.from_numpy(dummy_numpy)

# Warm up PyTorch
print("Warming up PyTorch baseline...")
for _ in range(5): 
    _ = pt_model(dummy_tensor, verbose=False)

# Benchmark PyTorch
print("Benchmarking PyTorch CPU...")
start = time.perf_counter()
runs = 30  # 30 runs is plenty to get a stable average on CPU
for _ in range(runs):
    _ = pt_model(dummy_tensor, verbose=False)
pytorch_time = (time.perf_counter() - start) / runs

# Warm up ONNX
print("Warming up ONNX Runtime...")
for _ in range(5): 
    _ = session.run(None, {input_name: dummy_numpy})

# Benchmark ONNX
print("Benchmarking ONNX Runtime CPU...")
start = time.perf_counter()
for _ in range(runs):
    _ = session.run(None, {input_name: dummy_numpy})
onnx_time = (time.perf_counter() - start) / runs

# Output Results
print("\n" + "="*40)
print("        CPU BENCHMARK RESULTS")
print("="*40)
print(f"PyTorch CPU Latency: {pytorch_time * 1000:.1f} ms ({1/pytorch_time:.1f} FPS)")
print(f"ONNX Runtime CPU Latency: {onnx_time * 1000:.1f} ms ({1/onnx_time:.1f} FPS)")

improvement = ((pytorch_time - onnx_time) / pytorch_time) * 100
print(f"Latency Reduction: {improvement:.1f}%")
print("="*40)