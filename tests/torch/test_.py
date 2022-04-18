import torch

class Test:

    def test(self):
        assert torch.cuda.is_available()
