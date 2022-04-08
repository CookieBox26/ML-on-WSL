from jax import grad
import jax.numpy as jnp
from pytest import approx


class Test:

    def test(self):
        # 勾配をとる
        g = grad(lambda x: jnp.exp(-x))
        assert g(0.0) == approx(-1.0, rel=1e-6)
        assert g(1.0) == approx(-0.367879, rel=1e-5)  # e の逆数
