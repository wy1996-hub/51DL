import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(df, x, alpha=0.01, iterations=100, epsilon=1e-8):
    history=[x]
    for i in range(iterations):
        if np.max(abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        x = x - alpha * df(x)
        history.append(x)
    return history

# 函数调用示例
df = lambda x: 3*x**2 - 6*x - 9
path = gradient_descent(df, 1., 0.01, 200)
print(path[-1])
# 绘制用箭头符号代表方向的线
f = lambda x: np.power(x, 3) - 3*x**2 - 9*x + 2
x = np.arange(-3, 4, 0.01)
y = f(x)
plt.plot(x, y)
path_x = np.asarray(path)
path_y = f(path_x)
plt.quiver(path_x[:-1], path_y[:-1], path_x[1:]-path_x[:-1],
           path_y[1:] - path_y[:-1], scale_units='xy', angles='xy', scale=1, color='k')
plt.scatter(path[-1], f(path[-1]))
plt.show()

