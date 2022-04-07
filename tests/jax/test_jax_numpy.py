from jax import grad
import jax.numpy as jnp

class TestJaxNumpy:
    def test(self):
        g = grad(lambda x: jnp.exp(-x))
        print(g)

