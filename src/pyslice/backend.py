# backend.py
import numpy as np
#import torch


def device_and_precision(device_spec=None):
    
    # We always choose PyTorch if available
    if xp != np:
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
    TORCH_AVAILABLE = True

except ImportError:
    xp = np
    config = device_and_precision()
    TORCH_AVAILABLE = False

DEFAULT_DEVICE, DEFAULT_FLOAT_DTYPE, DEFAULT_COMPLEX_DTYPE = config
del config

# Aliases for convenience
float_dtype = DEFAULT_FLOAT_DTYPE
complex_dtype = DEFAULT_COMPLEX_DTYPE


def configure_backend(device_spec=None, backend_spec=None):
    """
    Configure and return backend settings.

    Args:
        device_spec: Device specification ('cpu', 'cuda', 'mps', or None for auto)
        backend_spec: Backend specification ('numpy', 'torch', or None for auto)

    Returns:
        Tuple of (backend, device, float_dtype, complex_dtype)
    """
    global xp, DEFAULT_DEVICE, DEFAULT_FLOAT_DTYPE, DEFAULT_COMPLEX_DTYPE
    global float_dtype, complex_dtype

    # Determine backend
    if backend_spec == 'numpy':
        backend = np
        device = None
        fdtype = np.float64
        cdtype = np.complex128
    else:
        # Use torch
        backend = torch
        if device_spec is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_spec)

        # Set precision based on device
        if device.type == 'mps':
            fdtype = torch.float32
            cdtype = torch.complex64
        else:
            fdtype = torch.float64
            cdtype = torch.complex128

    return (backend, device, fdtype, cdtype)


def asarray(arraylike, dtype=None, device=None):
    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE
    if device is None:
        device = DEFAULT_DEVICE
    if xp != np:
#        if dtype == bool:
#            dtype = xp.bool
        array = xp.tensor(arraylike, dtype=dtype, device=device)
    else:
        array = xp.asarray(arraylike, dtype=dtype)
    return array

def zeros(dims, dtype=DEFAULT_FLOAT_DTYPE, device=DEFAULT_DEVICE):
    if xp != np:
        array = xp.zeros(dims, dtype=dtype, device=device)
    else:
        array = xp.zeros(dims, dtype=dtype)
    return array

def fftfreq(n, d, dtype=DEFAULT_FLOAT_DTYPE, device=DEFAULT_DEVICE):
    if xp != np:
        return xp.fft.fftfreq(n, d, dtype=dtype, device=device)
    else:
        return xp.fft.fftfreq(n, d, dtype=dtype)


def exp(x):
    return xp.exp(x)

def fft(k,**kwargs):
    if TORCH_AVAILABLE and "axis" in kwargs.keys():
        kwargs["dim"]=kwargs["axis"] ; del kwargs["axis"]
    if not TORCH_AVAILABLE and "dim" in kwargs.keys():
        kwargs["axis"]=kwargs["dim"] ; del kwargs["dim"]
    return xp.fft.fft(k,**kwargs)

def fftshift(k,**kwargs):
    if TORCH_AVAILABLE and "axes" in kwargs.keys():
        kwargs["dim"]=kwargs["axes"] ; del kwargs["axes"]
    if not TORCH_AVAILABLE and "dim" in kwargs.keys():
        kwargs["axes"]=kwargs["dim"] ; del kwargs["dim"]
    return xp.fft.fftshift(k,**kwargs)

def mean(k,**kwargs):
    if TORCH_AVAILABLE and "keepdims" in kwargs.keys():
        kwargs["keepdim"]=kwargs["keepdims"] ; del kwargs["keepdims"]
    if not TORCH_AVAILABLE and "keepdim" in kwargs.keys():
        kwargs["keepdims"]=kwargs["keepdim"] ; del kwargs["keepdim"]
    if TORCH_AVAILABLE and "axis" in kwargs.keys():
        kwargs["dim"]=kwargs["axis"] ; del kwargs["axis"]
    if not TORCH_AVAILABLE and "dim" in kwargs.keys():
        kwargs["axis"]=kwargs["dim"] ; del kwargs["dim"]
    return xp.mean(k,**kwargs)


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
    if xp != np:
        return xp.sum(x, dim=axis, **kwargs)
    else:
        return xp.sum(x, axis=axis, **kwargs)

def any(x):
    return xp.any(x)

def einsum(subscripts, *operands, **kwargs):
    if xp != np:
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
