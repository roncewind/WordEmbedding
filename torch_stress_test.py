import time

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# allocate large tensor and repeatedly perfor operations
try:
    print(f"Starting GPU stress loop using {device}...")
    for i in range(10000):
        a = torch.randn((1024, 1024), device=device)
        b = torch.randn((1024, 1024), device=device)
        for _ in range(100):
            c = torch.mm(a, b)
        if i % 100 == 0:
            print(f"Step {i}: {torch.cuda.memory_allocated() / 1e6:.2f} MB used")
        time.sleep(0.1)
except Exception as e:
    print(f"Error: {e}")
