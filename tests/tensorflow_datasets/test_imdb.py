import tensorflow_datasets as tfds


class TestImdb:

    def test(self):
        """
        注意： このテストを実行すると以下に IMDB データセットがロードされる
        ~/tensorflow_datasets/imdb_reviews/
        """
        data = tfds.load('imdb_reviews')
        assert len(data['train']) == 25000
        assert len(data['test']) == 25000

        data_train = data['train'].take(5)

        vocabulary = set()
        tokenizer = tfds.deprecated.text.Tokenizer()
        for x in data_train:
            vocabulary.update(tokenizer.tokenize(x['text'].numpy()))

        encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary)
        for x in data_train:
            encoded = encoder.encode(x['text'].numpy())
            assert type(encoded) == list  # 単語インデックスのリスト
