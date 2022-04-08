import trax
from trax.fastmath import numpy as fastnp
trax.fastmath.use_backend('jax')


class Test:
    def test(self):
        mat  = fastnp.array([[1, 2], [3, 4]])
