import paddle.fluid as fluid

# 定义卷积操作
def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups, # 过滤器个数
            conv_filter_size=3,                    # 过滤器大小
            conv_act='relu',
            conv_with_batchnorm=True,              # 表示在 Conv2d Layer 之后是否使用 BatchNorm
            conv_batchnorm_drop_rate=dropouts,     #表示 BatchNorm 之后的 Dropout Layer 的丢弃概率
            pool_type='max')                       # 最大池化

# CNN网络
def convolutional_neural_network(img):
    # 第一个卷积-池化层
    conv1 = fluid.layers.conv2d(input=img,     # 输入图像
                              num_filters=20,  # 卷积核数量，它与输出的通道相同
                              filter_size=5,   # 卷积核大小
                              act="relu")      # 激活函数
    pool1 = fluid.layers.pool2d(
              input=conv1,                     # 输入
              pool_size=2,                     # 池化核大小
              pool_type='max',                 # 池化类型
              pool_stride=2)                   # 池化步长
    conv_pool_1 = fluid.layers.batch_norm(pool1)
    # 第二个卷积-池化层
    conv2 = fluid.layers.conv2d(input=conv_pool_1,
                              num_filters=50,
                              filter_size=5,
                              act="relu")
    pool2 = fluid.layers.pool2d(
              input=conv2,
              pool_size=2,
              pool_type='max',
              pool_stride=2,
              global_pooling=False)
    conv_pool_2 = fluid.layers.batch_norm(pool2)
    # 第三个卷积-池化层
    conv3 = fluid.layers.conv2d(input=conv_pool_2,
                               num_filters=50,
                               filter_size=5,
                               act="relu")
    pool3 = fluid.layers.pool2d(
              input=conv3,
              pool_size=2,
              pool_type='max',
              pool_stride=2,
              global_pooling=False)
    # 全连接层
    fc = fluid.layers.fc(input=pool3, size=5, act=None)
    # 以softmax为激活函数的全连接输出层
    prediction = fluid.layers.fc(input=fc,
                                 size=5,
                                 act='softmax')
    return prediction

image = fluid.data(name='image', shape=[None,3, 47, 47], dtype='float32')   #[3, 47, 47]，表示为三通道，47*47的RGB图
label = fluid.data(name='label', shape=[None,1], dtype='int64')

predict = convolutional_neural_network(image)

# 损失函数
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(cost)
accuracy = fluid.layers.accuracy(input=predict, label=label)

test_program = fluid.default_main_program().clone(for_test=True)

# 参数优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
# Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计该函数实现了自适应矩估计优化器
optimizer.minimize(avg_cost)