import tensorflow
import tensorflow_text as tf_text


class Test:

    def test(self):
        t = tf_text.normalize_utf8(['１２３'], 'NFC')
        assert t.numpy()[0] == '１２３'.encode('utf-8')
        t = tf_text.normalize_utf8(['１２３'], 'NFKC')
        assert t.numpy()[0] == '123'.encode('utf-8')
