import numpy as np
import paddle
import paddle.nn as nn

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x):
        if self.drop_prob==0. or not self.training: # if prob is 0 or eval mode, return original input
            return x
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (x.shape[0], ) + (1, )*(x.ndim-1) # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        random_tensor = random_tensor.floor() # mask
        x = x.divide(keep_prob)*random_tensor # divide is to keep same output expectation
        return x

    def forward(self, inputs):
        return self.drop_path(inputs)