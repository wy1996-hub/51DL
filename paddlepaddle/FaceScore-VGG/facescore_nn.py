import paddle.fluid as fluid

# step2、配置网络

# 1.首先定义了一组卷积网络，即conv_block。卷积核大小为3x3，池化窗口大小为2x2，窗口滑动大小为2，groups决定每组VGG模块是几次连续的卷积操作，dropouts指定Dropout操作的概率。所使用的img_conv_group是在paddle.networks中预定义的模块，由若干组 Conv->BN->ReLu->Dropout 和一组 Pooling 组成。
# 2.五组卷积操作，即 5个conv_block。 第一、二组采用两次连续的卷积操作。第三、四、五组采用三次连续的卷积操作。每组最后一个卷积后面Dropout概率为0，即不使用Dropout操作。
# 3.最后接两层512维的全连接。
# 4.通过上面VGG网络提取高层特征，然后经过全连接层映射到类别维度大小的向量，再通过Softmax归一化得到每个类别的概率，也可称作分类器。

# VGG
def vgg_bn_drop(image, type_size):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,  # 具有[N，C，H，W]格式的输入图像
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,  # 过滤器个数
            conv_filter_size=3,  # 过滤器大小
            conv_act='relu',
            conv_with_batchnorm=True,  # 表示在 Conv2d Layer 之后是否使用 BatchNorm
            conv_batchnorm_drop_rate=dropouts,  # 表示 BatchNorm 之后的 Dropout Layer 的丢弃概率
            pool_type='max')  # 最大池化

    conv1 = conv_block(image, 64, 2, [0.0, 0])
    conv2 = conv_block(conv1, 128, 2, [0.0, 0.0])
    conv3 = conv_block(conv2, 256, 3, [0.0, 0.0, 0.0])
    conv4 = conv_block(conv3, 512, 3, [0.0, 0.0, 0.0])
    conv5 = conv_block(conv4, 512, 3, [0.0, 0.0, 0.0])

    drop = fluid.layers.dropout(x=conv2, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)

    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=1024, act=None)
    predict = fluid.layers.fc(input=fc2, size=type_size, act='softmax')
    return predict

#定义两个张量
image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

#获得分类器
predict=vgg_bn_drop(image,5)

# 定义损失函数、优化方法
# 交叉熵损失函数在分类任务上比较常用。
# 定义了一个损失函数之后，还要对它求平均值，因此定义的是一个Batch的损失值。
# 同时我们还可以定义一个准确率函数，这个可以在我们训练的时候输出分类的准确率。

# 定义损失函数
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(cost)
accuracy = fluid.layers.accuracy(input=predict, label=label)

# 克隆main_program得到test_program，使用参数for_test来区分该程序是用来训练还是用来测试，该api请在optimization之前使用.
test_program = fluid.default_main_program().clone(for_test=True)

# 优化方法
optimizer = fluid.optimizer.Adam(learning_rate=0.001)    # Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计该函数实现了自适应矩估计优化器
optimizer.minimize(avg_cost)
# use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# 创建一个Executor实例exe
exe = fluid.Executor(place)
# 正式进行网络训练前，需先执行参数初始化
exe.run(fluid.default_startup_program())

# 定义数据映射器
# DataFeeder负责将数据提供器（train_reader,test_reader）返回的数据转成一种特殊的数据结构，使其可以输入到Executor中。
# feed_list设置向模型输入的向变量表或者变量表名

# 数据映射器
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)