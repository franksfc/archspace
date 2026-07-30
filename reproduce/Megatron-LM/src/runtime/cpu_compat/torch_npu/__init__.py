"""CPU-only import sentinel for Open-Instruct data conversion.

This module is put first on ``PYTHONPATH`` only by the Dolci preprocessing
launcher.  Accelerate probes for ``torch_npu`` by importing it even when
PyTorch device-backend autoloading is disabled.  Providing an inert module
lets that probe complete; because CPU PyTorch has no ``torch.npu`` attribute,
Accelerate correctly reports that NPU execution is unavailable.

Training launchers never add this directory to ``PYTHONPATH``.
"""

