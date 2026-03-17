import os
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available:  {torch.cuda.is_available()}")
print(f"Device count:    {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Device name:     {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    print(f"Memory reserved:  {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")

    # Quick compute test on GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    print(f"\nGPU compute test: OK (matmul 1000x1000)")
    print(f"Memory after test: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
else:
    print("\nNo GPU detected. Check your ROCm installation.")

print("\nAll tests passed. Press Ctrl+C to exit (ROCm may hang on shutdown).")
torch.cuda.synchronize()  # Ensure all GPU work is done before exiting