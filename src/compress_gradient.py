import numpy as np
import blosc

import time
import sys

def compress(grad):
    assert isinstance(grad, np.ndarray)
    compressed_grad = blosc.pack_array(grad, cname='snappy')
    return compressed_grad

def decompress(msg):
    if sys.version_info[0] < 3:
        # Python 2.x implementation
        assert isinstance(msg, str)
    else:
        # Python 3.x implementation
        assert isinstance(msg, bytes)
    grad = blosc.unpack_array(msg)
    return grad