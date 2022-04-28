import torch
from reformer_pytorch import Reformer


class Test:

    def test(self):
        model = Reformer(
            dim=16,
            depth=4,
            heads=2,
            lsh_dropout=0.0,
            causal=True
        ).cuda()
        x = torch.randn(3, 128, 16).cuda()
        y = model(x)
        assert list(y.shape) == [3, 128, 16]
