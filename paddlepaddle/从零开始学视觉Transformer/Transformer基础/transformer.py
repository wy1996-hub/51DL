import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image

paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class PatchEmbedding(nn.Layer):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super().__init__()

        self.patch_embed = nn.Conv2D(in_channels,
                                      embed_dim,
                                      patch_size,
                                      patch_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [1,1,28,28] -> [n,embed_dim,h,w]
        x = self.patch_embed(x)
        # [n,embed_dim,h,w] -> [n,embed_dim,h*w]
        x = x.flatten(2)
        # [n,embed_dim,h*w] -> [n,h*w,embed_dim]
        x = x.transpose([0, 2, 1])
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

class Encoder(nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()

        self.atten = Identity() # TODO
        self.atten_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.atten_norm(x)
        x = self.atten(x)
        x = h+x
        h = x
        x = self.mlp(x)
        x = self.mlp_norm(x)
        x = h+x
        return x

class ViT(nn.Layer):
    def __init__(self):
        super().__init__()

        self.patch_embed = PatchEmbedding(224, 7, 3, 16)
        layer_list = [Encoder(16) for i in range(5)]
        self.encoders = nn.LayerList(layer_list)
        self.head = nn.Linear(16, 10)
        self.avgpool = nn.AdaptiveAvgPool1D(1)

    def forward(self, x):
        x = self.patch_embed(x)
        for encoder in self.encoders:
            x = encoder(x)
        # layernorm
        # [n,h*w,c] -> [n,c,h*w]
        x = x.transpose([0, 2, 1])
        # [n,c,h*w] -> [n, c, 1]
        x = self.avgpool(x)
        # [n, c, 1] -> [n, c]
        x = x.flatten(1)
        x = self.head(x)
        return x

def main():

    # # 1. load image and convert to tensor
    # img = np.array(Image.open('8.jpg').resize([28, 28]).convert('L'))
    # for i in range(28):
    #     for j in range(28):
    #         print(f'{img[i,j]}', end=' ')
    #     print()
    # sample = paddle.to_tensor(img, dtype='float32')
    # sample = sample.reshape([1, 1, 28, 28])
    # print(sample.shape)
    # # 2. patch embedding
    # patch_embed = PatchEmbedding(28, 7, 1, 512)
    # out = patch_embed(sample)
    # print(out.shape)
    # # 3. MLP 
    # mlp = MLP(512)
    # out = mlp(out)
    # print(out.shape)

    tensor = paddle.randn([4, 3, 224, 224])
    model = ViT()
    out = model(tensor)
    print(out.shape)


if __name__ == "__main__":
    main()