import trax


class TestReformer:
    def test(self):
        model = trax.models.Reformer(input_vocab_size=1000)
