# backend.py
import numpy as np
import torch


def configure_backend(device_spec=None, backend_spec=None):
    """Return (xp, device) based on device_spec string or None for default."""
    
    if backend_spec is None:
        if TORCH_AVAILABLE:
            xp = torch
        else:
            xp = np
    else:
        if backend_spec == 'torch':
            xp = torch
        elif backend_spec == 'numpy':
            xp = np
        else:
            raise NotImplementedError(f'Backend {backend_spec} is not Implemented.')

    # We always choose PyTorch if available
    if xp == torch:
        if device_spec is None:
            device = DEFAULT_DEVICE
        else: 
            device = xp.device(device_spec)
        xp.set_default_device(device)
    else:
        device = None 
    
    if device is not None and device.type == 'mps': # Use float32 for MPS (doesn't support float64), float64 for CPU/CUDA
        complex_dtype = xp.complex64
        float_dtype = xp.float32
    else:
        complex_dtype = xp.complex128
        float_dtype = xp.float64
    
    return xp, device, float_dtype, complex_dtype 



try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        xp, DEFAULT_DEVICE, float_dtype, complex_dtype = configure_backend('cuda')
    elif torch.backends.mps.is_available():
        xp, DEFAULT_DEVICE, float_dtype, complex_dtype = configure_backend('mps')
    else:
        xp, DEFAULT_DEVICE, float_dtype, complex_dtype = configure_backend('cpu')

except ImportError:
    TORCH_AVAILABLE = False
    DEFAULT_DEVICE = None
    xp, DEFAULT_DEVICE, float_dtype, complex_dtype = configure_backend()

