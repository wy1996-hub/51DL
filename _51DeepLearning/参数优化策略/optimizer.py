import numpy as np

# momentun动量法
# 更新向量不但考虑当前的梯度，还考虑上次更新的向量
def gradient_descent_momentum(df, x, alpha=0.01, gamma=0.8, iterations=100, epsilon=1e-8):
    history=[x]
    v = np.zeros_like(x) # 动量
    for i in range(iterations):
        if np.max(abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        v = gamma * v + alpha * df(x) # 更新动量
        x = x - v # 更新变量（参数）
        history.append(x)
    return history

# adagrad自适应梯度
# 将每个梯度分量除以该梯度分量的历史累加值
# 解决一个分量x1合适的学习率对另一个分量x2可能过大或过小，造成震荡或停滞
def gradient_descent_adagrad(df, x, alpha=0.01, gamma=0.8, iterations=100, epsilon=1e-8):
    history=[x]
    gl = np.ones_like(x)
    for i in range(iterations):
        if np.max(abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        gl += grad**2
        x = x - alpha * grad / (np.sqrt(gl) + epsilon)
        history.append(x)
    return history

# adadelta法
# 使用历史累加值的均方和代替直接的累加和
def gradient_descent_adadelta(df, x, alpha=0.01, rho=0.9, iterations=100, epsilon=1e-8):
    history=[x]
    Eg = np.ones_like(x)
    Edelta = np.ones_like(x)
    for i in range(iterations):
        if np.max(abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        Eg = rho * Eg + (1 - rho) * (grad**2)
        delta = np.sqrt((Edelta + epsilon) / (Eg + epsilon)) * grad
        x = x - alpha * delta
        Edelta = rho * Edelta + (1 - rho) * (delta**2)
        history.append(x)
    return history

# RMSprop法
# 将梯度除以其平均长度，即转化为单位长度的梯度
# 目的：总是以固定步长alpha更新参数x
def gradient_descent_RMSprop(df, x, alpha=0.01, beta=0.9, iterations=100, epsilon=1e-8):
    history=[x]
    v = np.ones_like(x)
    for i in range(iterations):
        if np.max(abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        v = beta * v + (1 - beta) * (grad**2)
        x = x - alpha * grad / (np.sqrt(v) + epsilon)
        history.append(x)
    return history

# Adam法
# 和rmsprop一样，存储一个指数衰减的过去梯度的平方的累积平均
# 和momentum一样，存储了梯度的累积平均
# 相当于梯度的一阶和二阶动量
def gradient_descent_adam(df, x, alpha=0.01, beta_1=0.9, beta_2=0.999, iterations=100, epsilon=1e-8):
    history=[x]
    m = np.ones_like(x)
    v = np.ones_like(x)
    for i in range(iterations):
        if np.max(abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * (grad**2)
        i = i + 1
        if True:
            m_1 = m / (1 - np.power(beta_1, i + 1))
            v_1 = v / (1 - np.power(beta_2, i + 1))
        else:
            m_1 = m / (1 - np.power(beta_1, i)) + (1 - beta_1) * grad / (1 - np.power(beta_1, i))
            v_1 = v / (1 - np.power(beta_2, i))
        x = x - alpha * m_1 / (np.sqrt(v_1) + epsilon)
        history.append(x)
    return history

