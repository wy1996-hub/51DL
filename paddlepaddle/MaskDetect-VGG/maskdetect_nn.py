import paddle.fluid as fluid

# 二、模型配置

class ConvPool(fluid.dygraph.Layer):
    '''卷积+池化'''
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 groups,
                 pool_padding=0,
                 pool_type='max',
                 conv_stride=1,
                 conv_padding=1,
                 act=None):
        super(ConvPool, self).__init__()

        self._conv2d_list = []

        for i in range(groups):
            conv2d = self.add_sublayer(    # 返回一个由所有子层组成的列表。
                'bb_%d' % i,
                fluid.dygraph.Conv2D(
                num_channels=num_channels, # 通道数
                num_filters=num_filters,   # 卷积核个数
                filter_size=filter_size,   # 卷积核大小
                stride=conv_stride,        # 步长
                padding=conv_padding,      # padding大小，默认为0----修改为默认1，否则shape对不上
                act=act)
            )
            num_channels = num_filters
            self._conv2d_list.append(conv2d)

        self._pool2d = fluid.dygraph.Pool2D(
            pool_size=pool_size,           # 池化核大小
            pool_type=pool_type,           # 池化类型，默认是最大池化
            pool_stride=pool_stride,       # 池化步长
            pool_padding=pool_padding      # 填充大小
            )

    def forward(self, inputs):
        x = inputs
        for conv in self._conv2d_list:
            x = conv(x)
            print(x.shape)
        x = self._pool2d(x)
        return x


class VGGNet(fluid.dygraph.Layer):
    '''VGG网络'''
    def __init__(self):
        super(VGGNet, self).__init__()

        self.convpool01 = ConvPool(3, 64, 3, 2, 2, 2, act='relu')
        self.convpool02 = ConvPool(64, 128, 3, 2, 2, 2, act='relu')
        self.convpool03 = ConvPool(128, 256, 3, 2, 2, 3, act='relu')
        self.convpool04 = ConvPool(256, 512, 3, 2, 2, 3, act='relu')
        self.convpool05 = ConvPool(512, 512, 3, 2, 2, 3, act='relu')

        self.pool_5_shape = 512 * 7 * 7
        self.fc01 = fluid.dygraph.Linear(self.pool_5_shape, 4096, act='relu')
        self.fc02 = fluid.dygraph.Linear(4096, 4096, act='relu')
        self.fc03 = fluid.dygraph.Linear(4096, 2, act='softmax')

    def forward(self, inputs, label=None):      # [3,224,224]
        """前向计算"""
        out = self.convpool01(inputs)           # [8,64,112,112]
        out = self.convpool02(out)              # [8,128,56,56]
        out = self.convpool03(out)              # [8,256, 28,28]
        out = self.convpool04(out)              # [8,512,14,14]
        out = self.convpool05(out)              # [8,512,7,7]

        out = fluid.layers.reshape(out, shape=[-1, 512*7*7])
        out = self.fc01(out)
        out = self.fc02(out)
        out = self.fc03(out)

        if label is not None:
            acc = fluid.layers.accuracy(input=out, label=label)
            return out, acc
        else:
            return out