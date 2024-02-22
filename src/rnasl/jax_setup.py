import os

os.environ["JAX_ENABLE_X64"] = "True"

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

DTYPE = jnp.float32
jfloat = jnp.float32 if DTYPE == jnp.float32 else jnp.float64

import jax
def get_preferred_device():
    platform = os.environ.get("JAX_PLATFORM_NAME", None)

    if platform == "cpu":
        print("Forcing CPU usage")
        return jax.devices("cpu")[0]

    elif platform == "gpu":
        try:
            gpus = jax.devices("gpu")
            if gpus:
                print("Using GPU")
                return gpus[0]
            else:
                raise RuntimeError("GPU requested but not available.")
        except RuntimeError as e:
            raise RuntimeError("GPU backend requested but not available in current JAX install.") from e

    else:
        try:
            gpus = jax.devices("gpu")
            if gpus:
                print("Defaulting to GPU")
                return gpus[0]
        except RuntimeError:
            pass  # GPU backend not available

        print("No GPU found or JAX GPU backend not available, using CPU")
        return jax.devices("cpu")[0]
