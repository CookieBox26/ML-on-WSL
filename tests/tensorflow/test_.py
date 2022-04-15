from tensorflow.python.client import device_lib


class Test:

    def test(self):
        devices = device_lib.list_local_devices()
        assert len(devices) == 2
        assert devices[0].name == '/device:CPU:0'
        assert devices[1].name == '/device:GPU:0'

