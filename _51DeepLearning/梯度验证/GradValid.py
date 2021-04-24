import numpy as np

# 通用的数值梯度验证方法
def numerical_gradient(f, params, epsilon=1e-6):# 参数一传入的是函数对象，而非函数值
    numerical_grads = []
    for x in params:# x可能是一个多维数组，对其中的每一个元素都要计算他对应的偏导数
        grad = np.zeros(x.shape)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_value = x[idx]
            x[idx] = old_value + epsilon
            fx = f()
            x[idx] = old_value - epsilon
            fx_ = f()
            grad[idx] = (fx - fx_) / (2 * epsilon)
            x[idx] = old_value# 要将该权值参数恢复到原来的值
            it.iternext()
        numerical_grads.append(grad)
    return numerical_grads

