#!/usr/bin/env python3
"""
Patch vLLM for RDNA 4 container compatibility.

Three issues when running vLLM in ROCm containers on RDNA 4:
  1. amdsmi fails in containers (can't access sysfs/hwmon)
  2. Circular import in _get_gcn_arch() (GPU name detection fails)
  3. torch.cuda.device_count() returns 0 despite HIP working

This script creates vllm/rocm_patches/rdna4_init.py with HIP-based
workarounds and injects an import into vllm/__init__.py.

Based on vLLM issue #40081.
"""

import sys
from pathlib import Path


RDNA4_INIT_PY = '''\
"""RDNA 4 container compatibility patches for vLLM on ROCm."""
import ctypes
import torch.cuda


def _get_gpu_count_hip():
    """Fallback GPU count via HIP C API when torch fails to detect."""
    try:
        hip_lib = ctypes.CDLL("libamdhip64.so")
        count = ctypes.c_int()
        err = hip_lib.hipGetDeviceCount(ctypes.byref(count))
        if err == 0 and count.value > 0:
            return count.value
    except Exception:
        pass
    return 0


_original_device_count = torch.cuda.device_count


def _hip_device_count():
    """Override torch.cuda.device_count with HIP fallback."""
    count = _original_device_count()
    return count if count > 0 else _get_gpu_count_hip()


# Monkey-patch device count detection
torch.cuda.device_count = _hip_device_count
'''

INIT_INJECT = """
# RDNA 4 container compatibility
try:
    from vllm.rocm_patches.rdna4_init import *
except ImportError:
    pass
"""


def patch_vllm(vllm_root: str) -> None:
    root = Path(vllm_root)
    if not (root / "vllm").is_dir():
        print(f"ERROR: {vllm_root}/vllm/ not found", file=sys.stderr)
        sys.exit(1)

    # Create rocm_patches directory
    patches_dir = root / "vllm" / "rocm_patches"
    patches_dir.mkdir(exist_ok=True)
    (patches_dir / "__init__.py").touch()

    # Write rdna4_init.py
    init_file = patches_dir / "rdna4_init.py"
    init_file.write_text(RDNA4_INIT_PY)
    print(f"Created {init_file}")

    # Inject import into vllm/__init__.py
    vllm_init = root / "vllm" / "__init__.py"
    content = vllm_init.read_text()
    if "rdna4_init" not in content:
        content += INIT_INJECT
        vllm_init.write_text(content)
        print(f"Patched {vllm_init} with RDNA 4 import")
    else:
        print(f"{vllm_init} already patched")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/app/vllm"
    patch_vllm(target)
