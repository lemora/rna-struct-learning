import jax
import jaxlib


def test_cuda():
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print("jaxlib version:", jaxlib.__version__)
    print()


if __name__ == "__main__":
    test_cuda()
