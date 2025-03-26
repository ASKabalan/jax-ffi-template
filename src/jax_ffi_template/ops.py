import jax
import jax.numpy as jnp
import jax_ffi_template_lib.gpu_ops as gpu_ops
import numpy as np

for name, fn in gpu_ops.registrations().items():
    jax.ffi.register_ffi_target(name, fn, platform='CUDA')


def add_element(a, scaler=1.0):
    if a.dtype != jnp.float32:
        raise ValueError('Only float32 is supported')
    if not isinstance(scaler, float):
        raise ValueError('Only float32 is supported')

    out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)

    return jax.ffi.ffi_call('add_element', out_type, vmap_method='sequential')(a, scaler=np.float32(scaler))
