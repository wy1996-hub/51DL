import paddle
import paddle.nn as nn
import DropPath

def _init_weights_linear():
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
    bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(.0))
    return weight_attr, bias_attr

def _init_weights_layernorm():
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
    bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(.0))
    return weight_attr, bias_attr

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ConvNormAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2D(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              groups=groups, 
                              bias_attr=bias_attr, 
                              weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()))
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = nn.Silu()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class MLP(nn.Layer):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.):
        super().__init__()
        w_attr1, b_attr1 = _init_weights_linear()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim*mlp_ratio), w_attr1, b_attr1)
        w_attr2, b_attr2 = _init_weights_linear()
        self.fc2 = nn.Linear(int(embed_dim*mlp_ratio), embed_dim, w_attr2, b_attr2)

        self.act = nn.Silu()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class Attention(nn.Layer):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 qkv_bias=True, 
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_dim = int(embed_dim // self.num_heads)
        self.all_head_dim = self.attn_head_dim * self.num_heads

        w_attr1, b_attr1 = _init_weights_linear()
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim*3, # weights for q, k, v
                             weight_attr=w_attr1, 
                             bias_attr=b_attr1 if qkv_bias else False)
        
        self.scales = self.attn_head_dim ** -0.5

        w_attr2, b_attr2 = _init_weights_linear()
        self.proj = nn.Linear(embed_dim,
                              embed_dim,
                              weight_attr=w_attr2,
                              bias_attr=b_attr2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)   
        self.softmax = nn.Softmax(axis=-1)

    def __transpose_multihead(self, x):
        # in_shape: [batch_size, P(patch_nums), N, H*D(all_head_dim\embed_dim)]
        B, P, N, HD = x.shape
        x = x.reshape([B, P, N, self.num_heads, -1])
        x =  x.transpose([0, 1, 3, 2, 4])
        # out_shape: [batch_size, P, num_heads, N, D(attn_head_dim)]
        return x

    def forward(self, x):
        # [B, 2x2, 256, 96]: [B, P, N, D]
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.__transpose_multihead, qkv)

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = attn*self.scales
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        # [batch_size, P, num_heads, N, N]

        z = paddle.matmul(attn, v)
        # [batch_size, P, num_heads, N, D]
        z =  z.transpose([0, 1, 3, 2, 4])
        B, P, N, H, D = z.shape
        z = z.reshape([B, P, N, H*D])
        z = self.proj(z)
        z = self.proj_dropout(z)
        return z

class EncoderLayer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 qkv_bias=True,
                 mlp_ratio=2.,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.): 
        super().__init__()
        w_attr1, b_attr1 = _init_weights_layernorm()
        w_attr2, b_attr2 = _init_weights_layernorm()

        self.attn_norm = nn.LayerNorm(embed_dim, weight_attr=w_attr1, bias_attr=b_attr1)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, dropout, attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0 else Identity()
        self.mlp_norm = nn.LayerNorm(embed_dim, weight_attr=w_attr2, bias_attr=b_attr2)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x

class Transformer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 qkv_bias=True,
                 mlp_ratio=2.,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]

        layer_list = []
        for i in range(depth):
            layer_list.append(EncoderLayer(embed_dim,
                                           num_heads,
                                           qkv_bias,
                                           mlp_ratio,
                                           dropout,
                                           attention_dropout))
        self.layers = nn.LayerList(layer_list)

        w_attr, b_attr = _init_weights_layernorm()
        self.norm = nn.LayerNorm(embed_dim, weight_attr=w_attr, bias_attr=b_attr)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class MobileV2Block(nn.Layer):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 stride=1, 
                 expansion=4):
        super().__init__()
        self.stride = stride # assert stride in [1, 2]

        hidden_dim = int(round(in_channels*expansion))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expansion != 1:
            layers.append(ConvNormAct(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            # dw:逐通道卷积
            ConvNormAct(hidden_dim, hidden_dim, stride=stride, padding=1, groups=hidden_dim),
            # pw:逐点卷积
            nn.Conv2D(hidden_dim, out_channels, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(out_channels)])

        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileViTBlock(nn.Layer):
    def __init__(self,
                 dim,
                 hidden_dim,
                 depth,
                 num_heads=8,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 patch_size=(2, 2)):
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        # local representations
        self.conv1 = ConvNormAct(dim, dim, padding=1)
        self.conv2 = ConvNormAct(dim, hidden_dim, kernel_size=1)

        # global representations
        self.transformer = Transformer(embed_dim=hidden_dim,
                                       num_heads=num_heads,
                                       depth=depth,
                                       qkv_bias=qkv_bias,
                                       mlp_ratio=mlp_ratio,
                                       dropout=dropout,
                                       attention_dropout=attention_dropout,
                                       droppath=droppath)

        # fusion
        self.conv3 = ConvNormAct(hidden_dim, dim, kernel_size=1)
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvNormAct(2 * dim, dim, padding=1)
    
    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.conv2(x)
        # [B, 96, 32, 32]

        B, C, H, W = x.shape
        x = x.reshape([B, C, H//self.patch_h, self.patch_w, W//self.patch_w, self.patch_w])
        # [4, 96, 16, 2, 16, 2]
        x = x.transpose([0, 1, 3, 5, 2, 4])
        # [4, 96, 2, 2, 16, 16]
        x = x.reshape([B, C, (self.patch_h * self.patch_w), -1]) #[B, C, ws**2, n_windows**2]
        x = x.transpose([0, 2, 3, 1]) #[B, ws**2, n_windows**2, C]
        # [4, 4, 256, 96]
        x = self.transformer(x)
        x = x.reshape([B, self.patch_h, self.patch_w, H//self.patch_h, W//self.patch_w, C])
        x = x.transpose([0, 5, 3, 1, 4, 2])
        x = x.reshape([B, C, H, W])

        x = self.conv3(x)
        x = paddle.concat((h, x), axis=1) # 通道维度concat
        x = self.conv4(x)
        return x

class MobileViT(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 dims=[16, 32, 48, 48, 48, 64, 80, 96, 384],
                 hidden_dims=[96, 120, 144], # d: hidden dims in mobilevit block
                 num_classes=1000):
        super().__init__()
        # [B, 3, 256, 256]
        self.conv3x3 = ConvNormAct(in_channels, dims[0], stride=2, padding=1)
        # [B, 16, 128, 128]
        self.mv2_block1 = MobileV2Block(dims[0], dims[1])
        # [B, 32, 128, 128]
        self.mv2_block2 = MobileV2Block(dims[1], dims[2], stride=2)
        # [B, 48, 64, 64]
        self.mv2_block3 = MobileV2Block(dims[2], dims[3])
        # [B, 48, 64, 64]
        self.mv2_block4 = MobileV2Block(dims[3], dims[4]) # repeat = 2
        # [B, 48, 64, 64]
        self.mv2_block5 = MobileV2Block(dims[4], dims[5], stride=2)
        # [B, 64, 32, 32]
        self.mvit_block1 = MobileViTBlock(dims[5], hidden_dims[0], depth=2)
        # [B, 64, 32, 32]
        self.mv2_block6 = MobileV2Block(dims[5], dims[6], stride=2)
        # [B, 80, 16, 16]
        self.mvit_block2 = MobileViTBlock(dims[6], hidden_dims[1], depth=4)
        # [B, 80, 16, 16]
        self.mv2_block7 = MobileV2Block(dims[6], dims[7], stride=2)
        # [B, 96, 8, 8]
        self.mvit_block3 = MobileViTBlock(dims[7], hidden_dims[2], depth=3)
        # [B, 96, 8, 8]
        self.conv1x1 = ConvNormAct(dims[7], dims[8], kernel_size=1) 
        # [B, 384, 8, 8]
        self.pool = nn.AdaptiveAvgPool2D(1) # 2D将两个维度都池化
        # [B, 384, 1, 1]
        self.linear = nn.Linear(dims[8], num_classes)
        # [B, 1000]

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.mv2_block1(x)

        x = self.mv2_block2(x)
        x = self.mv2_block3(x)
        x = self.mv2_block4(x)
        
        x = self.mv2_block5(x)
        x = self.mvit_block1(x)

        x = self.mv2_block6(x)
        x = self.mvit_block2(x)

        x = self.mv2_block7(x)
        x = self.mvit_block3(x)
        x = self.conv1x1(x)

        x = self.pool(x)
        x = x.reshape(x.shape[:2])
        x = self.linear(x)
        return x

# def build_mobile_vit(config):
#     """Build MobileViT by reading options in config object
#     Args:
#         config: config instance contains setting options
#     Returns:
#         model: MobileViT model
#     """
#     model = MobileViT(in_channels=config.MODEL.IN_CHANNELS,
#                       dims=config.MODEL.DIMS,  # XS: [16, 32, 48, 48, 48, 64, 80, 96, 384]
#                       hidden_dims=config.MODEL.HIDDEN_DIMS, # XS: [96, 120, 144], # d: hidden dims in mobilevit block
#                       num_classes=config.MODEL.NUM_CLASSES)
#     return model

def main():
   model = MobileViT()
   print(model)
   t = paddle.randn([4, 3, 256, 256])
   out = model(t)
   print(out.shape)

if __name__  == "__main__":
   main()