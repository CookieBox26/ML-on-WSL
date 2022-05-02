from typing import Sequence
import jax
import jax.numpy as jnp
from flax import linen


class MLP(linen.Module):
    features: Sequence[int]

    @linen.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = linen.relu(linen.Dense(feat)(x))
        x = linen.Dense(self.features[-1])(x)
        return x


class TestFlaxLinen:
    def test(self):
        key = jax.random.PRNGKey(26)

    def test_mlp(self):
        model = MLP([12, 8, 4])
        batch = jnp.ones((32, 10))
        variables = model.init(jax.random.PRNGKey(0), batch)
        output = model.apply(variables, batch)
