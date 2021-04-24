import numpy as np

# 前面的参数优化器是将变量X（参数）的优化策略硬编码在了梯度下降算法中（例如gradient_descent_momentum）
# 不同优化策略的梯度下降法除了参数更新不同外，其梯度下降算法的框架是一样的
# 为提高代码的复用性和灵活性，将参数的优化策略从梯度下降算法中分离出来
# 可以用不同的类来表示不同的参数优化方法--参数优化器

# 基类
class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self, grads):
        pass

    def parameters(self):
        return self.params

# 子类
class Opt_Basic(Optimizer):
    def __init__(self, params, learning_rate):
        super().__init__(params)
        self.learning_rate = learning_rate

    def step(self, grads):# 迭代更新参数
        for i in range(len(self.params)):
            self.params[i] -= self.learning_rate * grads[i]
        return self.params

class Opt_Momentum(Optimizer):
    def __init__(self, params, learning_rate, gamma):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.v = []
        for param in params:
            self.v.append(np.zeros_like(param))

    def step(self, grads):
        for i in range(len(self.params)):
            self.v[i] = self.gamma * self.v[i] + self.learning_rate * grads[i]
            self.params[i] -= self.v[i]
        return self.params

# 分离后的接收参数优化器的梯度下降算法
def gradient_descent_(df, optimizer, iterations, epsilon=1e-8):
    x,  = optimizer.parameters()
    x = x.copy()
    history = [x]
    for i in range(iterations):
        if np.max(abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        x,  = optimizer.step([grad])
        x = x.copy()
        history.append(x)
    return history

# 函数调用示例1
df = lambda x: np.array(((1/8)*x[0], 18*x[1]))
x0 = np.array([-2.4, 0.2])
optimizator = Opt_Basic([x0], 0.1)# 区别就在于调用的优化器实例不同
path = gradient_descent_(df, optimizator, 1000)
print(path[-1])
path = np.asarray(path)
path = path.transpose()

# 函数调用示例2
# df = lambda x: np.array(((1/8)*x[0], 18*x[1]))
# x0 = np.array([-2.4, 0.2])
# optimizator = Opt_Momentum([x0], 0.1, 0.8)
# path = gradient_descent_(df, optimizator, 1000)
# print(path[-1])
# path = np.asarray(path)
# path = path.transpose()