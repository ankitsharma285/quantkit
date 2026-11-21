import os

# Without a GPU, run Triton kernels in its CPU interpreter so kernel tests still execute.
# Must be set before triton is imported anywhere.
try:
    import torch

    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
except ImportError:
    pass
