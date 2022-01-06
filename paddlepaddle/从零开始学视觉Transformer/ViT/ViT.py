# Author: 51
# 2021.12.1

import paddle
import paddle.nn as nn
import copy

paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim/num_heads)
        self.all_head_dim = self.head_dim*num_heads
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim*3,
                             bias_attr=False if qkv_bias is False else None)
        self.scale = self.head_dim**-0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(axis=-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        # [B patch_nums all_head_dim] -> [B patch_nums num_heads head_dim]
        x = x.reshape(new_shape)
        # [B patch_nums num_heads head_dim] -> [B num_heads patch_nums head_dim]
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        B, N, _ = x.shape # # [B patch_nums embed_dim]

        # [B patch_nums embed_dim] -> [B patch_nums all_head_dim*3]
        qkv = self.qkv(x)
        # [B patch_nums all_head_dim*3] -> [B patch_nums all_head_dim]*3
        qkv = qkv.chunk(3, -1)
        # 分三次输入函数 q=akv[0] k=qkv[1] v=qkv[2]
        q, k, v = map(self.transpose_multi_head, qkv)
        
        # q, k, v: [B num_heads patch_nums head_dim] 对tensor末尾通道的数据进行矩阵乘法
        attn = paddle.matmul(q, k, transpose_y=True)
        attn *= self.scale
        attn = self.softmax(attn) # attn: [B num_heads patch_nums patch_nums] 每一个patch对其他所有patch的attention
        attn = self.attention_dropout(attn)
        
        # [B num_heads patch_nums patch_nums] * [B num_heads patch_nums head_dim] -> [B num_heads patch_nums head_dim]
        out = paddle.matmul(attn, v)
        # [B num_heads patch_nums head_dim] -> [B patch_nums num_heads head_dim]
        out = out.transpose([0, 2, 1, 3])

        # [B patch_nums num_heads head_dim] -> # [B patch_nums all_head_dim]
        out = out.reshape([B, N, -1])
        # [B patch_nums all_head_dim] -> [B patch_nums embed_dim]
        out = self.proj(out)
        out = self.dropout(out)

        return out

class PatchEmbedding(nn.Layer):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.):
        super().__init__()

        n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_embedding = nn.Conv2D(in_channels,
                                      embed_dim,
                                      patch_size,
                                      patch_size)
        self.dropout = nn.Dropout(dropout)
        self.class_token = paddle.create_parameter(shape=[1, 1, embed_dim],
                                                   dtype='float32', 
                                                   default_initializer=nn.initializer.Constant(0.))
        self.position_embedding = paddle.create_parameter(shape=[1, n_patches+1, embed_dim], 
                                                          dtype='float32', 
                                                          default_initializer=nn.initializer.TruncatedNormal(std=.02))

    def forward(self, x):
        class_tokens = self.class_token.expand([x.shape[0], -1, -1]) # 给batch中的每一张图片添加一个class_token
        # [N C H W] -> [n,embed_dim,h,w]
        x = self.patch_embedding(x)
        # [n,embed_dim,h,w] -> [n,embed_dim,h*w]
        x = x.flatten(2)
        # [n,embed_dim,h*w] -> [n,h*w,embed_dim]
        x = x.transpose([0, 2, 1])
        x = paddle.concat([class_tokens, x], axis=1)
        x += self.position_embedding
        x = self.dropout(x)
        return x

class MLP(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, int(embed_dim*mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim*mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Layer):
    def __init__(self, embed_dim=768, num_heads=4, qkv_bias=True, mlp_ratio=4., dropout=0.):
        super().__init__()

        self.atten = Attention(embed_dim, num_heads)
        self.atten_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.atten_norm(x) # pre norm
        x = self.atten(x)
        x = h+x
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h+x
        return x

class Encoder(nn.Layer):
    def __init__(self, embed_dim, depth):
        super().__init__()

        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer()
            layer_list.append(encoder_layer)
        self.layers = nn.LayerList(layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class ViT(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=5,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.encoder = Encoder(embed_dim, depth)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.classifier(x[:, 0])
        return x

def main():
    model = ViT()
    print(model)
    paddle.summary(model, (4, 3, 224, 224))

if __name__ == "__main__":
    main()