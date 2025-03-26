import jax.numpy as jnp
from numpy.testing import assert_allclose

import jax_ffi_template


def test_axpy():
    a = jnp.array([1, 2, 3], dtype=jnp.float32)

    # Add 1 to each element of the array
    b = jax_ffi_template.ops.add_element(a, scaler=1.5)

    c = a + 1.5

    assert_allclose(b, c)
    print(f'Original array: {a}')
    print(f'Array after adding 1 to each element With cuda code: {b}')
    print(f'Array after adding 1 to each element With XLA: {c}')
