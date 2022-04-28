from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class TestFlaxLinen:
    def test_mlp(self):
        model = MLP([12, 8, 4])
        batch = jnp.ones((32, 10))
        variables = model.init(jax.random.PRNGKey(0), batch)
        output = model.apply(variables, batch)