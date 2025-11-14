# backend.py
import numpy as np
import torch


def device_and_precision(device_spec=None):
    
    # We always choose PyTorch if available
    if xp == torch:
        if device_spec is None:
            device = DEFAULT_DEVICE
        else: 
            device = xp.device(device_spec)
    else:
        device = None
    
    if device is not None and device.type == 'mps': # Use float32 for MPS (doesn't support float64), float64 for CPU/CUDA
        complex_dtype = xp.complex64
        float_dtype = xp.float32
    else:
        complex_dtype = xp.complex128
        float_dtype = xp.float64
    
    return device, float_dtype, complex_dtype 


try:
    import torch
    xp = torch
    if torch.cuda.is_available():
        config = device_and_precision('cuda')
    elif torch.backends.mps.is_available():
        config = device_and_precision('mps')
    else:
        config = device_and_precision('cpu')

except ImportError:
    xp = np
    config = device_and_precision()

DEFAULT_DEVICE, DEFAULT_FLOAT_DTYPE, DEFAULT_COMPLEX_DTYPE = config
del config


def asarray(arraylike, dtype=None, device=None):
    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE
    if device is None:
        device = DEFAULT_DEVICE
    if xp == torch:
#        if dtype == bool:
#            dtype = xp.bool
        array = xp.tensor(arraylike, dtype=dtype, device=device)
    else:
        array = xp.asarray(arraylike, dtype=dtype)
    return array

def zeros(dims, dtype=DEFAULT_FLOAT_DTYPE, device=DEFAULT_DEVICE):
    if xp == torch:
        array = xp.zeros(dims, dtype=dtype, device=device)
    else:
        array = xp.zeros(dims, dtype=dtype)
    return array

def fftfreq(n, d, dtype=DEFAULT_FLOAT_DTYPE, device=DEFAULT_DEVICE):
    if xp is torch:
        return xp.fft.fftfreq(n, d, dtype=dtype, device=device)
    else:
        return xp.fft.fftfreq(n, d, dtype=dtype)


def exp(x):
    return xp.exp(x)

def fft(k):
    return xp.fft.fft(k)

def ifft2(k):
    return xp.fft.ifft2(k)

def real(x):
    return xp.real(x)

def absolute(x):
    return xp.absolute(x)

def amax(x):
    return xp.amax(x)

def amin(x):
    return xp.amin(x)

def sum(x, axis=None, **kwargs):
    if xp is torch:
        return xp.sum(x, dim=axis, **kwargs)
    else:
        return xp.sum(x, axis=axis, **kwargs)

def any(x):
    return xp.any(x)

def einsum(subscripts, *operands, **kwargs):
    if xp is torch:
        return xp.einsum(subscripts, *operands, **kwargs)
    else:
        return xp.einsum(subscripts, *operands, optimize=True, **kwargs)

def to_cpu(array):
    if type(array) == np.ndarray:
        return array
    else:
        return array.cpu().numpy()

def isnan(x):
    return xp.isnan(x)
