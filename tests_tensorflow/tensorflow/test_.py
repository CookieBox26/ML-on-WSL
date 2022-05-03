from tensorflow.python.client import device_lib
import tensorflow as tf


class TestTensorFlow:

    def test_device_lib(self):
        devices = device_lib.list_local_devices()
        assert len(devices) == 2
        assert devices[0].name == '/device:CPU:0'
        assert devices[1].name == '/device:GPU:0'

    def test_ensure_shape(self):
        x = tf.constant([[1, 2, 3], [4, 5, 6]])
        x = tf.ensure_shape(x, [None, 3])
        x = tf.ensure_shape(x, [2, None])
        x = tf.ensure_shape(x, [2, 3])
