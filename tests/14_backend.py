from pyslice.backend import xp, DEFAULT_DEVICE, float_dtype, complex_dtype
from pyslice.backend import configure_backend

print(xp, DEFAULT_DEVICE, float_dtype, complex_dtype)

print(configure_backend('cuda'))
print(configure_backend('cpu'))
print(configure_backend('mps'))
print(configure_backend('cpu', 'numpy'))
print(configure_backend('cuda', 'numpy'))
print(configure_backend(backend_spec='numpy'))


