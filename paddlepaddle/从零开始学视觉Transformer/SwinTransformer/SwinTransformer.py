import paddle
import paddle.nn as nn

class PatchEmbedding(nn.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_embed = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x) # [B embed_dim h' w']
        x = x.flatten(2) # [B embed_dim h'*W']
        x = x.transpose([0, 2, 1]) # [B h'*W' embed_dim]
        x = self.norm(x) 
        return x

class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.norm = nn.LayerNorm(dim*4)
        self.reduction = nn.Linear(dim*4, dim*2)

    def forward(self, x):
        b, _, c = x.shape
        h, w = self.resolution
        x = x.reshape([b, h, w, c])

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = paddle.concat([x0, x1, x2, x3], axis=-1) # axis=-1, 拼接最后一个轴，dim轴 -> [b, h//2, w//2, 4c]
        x = x.reshape([b, -1, 4*c])
        x = self.norm(x)
        x = self.reduction(x)
        return x

class MLP(nn.Layer):
    def __init__(self, dim, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim*mlp_ratio))
        self.fc2 = nn.Linear(int(dim*mlp_ratio), dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

def windows_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.reshape([b, h//window_size, window_size, w//window_size, window_size, c])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([-1, window_size, window_size, c])
    return x

def windows_reverse(x, window_size, h, w):
    b = int(x.shape[0] // ((h/window_size)*(w/window_size)))
    x = x.reshape([b, h//window_size, w//window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([b, h, w, -1])
    return x

class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim//self.num_heads       
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3)       
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(dim, dim)

    def transpose_multi_head(self, x):
        # x: [B num_patches=h*w c] 但是x[:-1]代表最后一个通道，并未取到
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim] # [B num_patches num_heads head_dim] 
        # [B num_patches c] -> [B num_patches num_heads head_dim] 
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3]) # [B num_heads num_patches head_dim] 深度学习都是对通道维度进行处理的，需把通道维度转移到最后一位
        return x

    def forward(self, x):
        b, n, c = x.shape # [B num_patches=h*w c]
        qkv = self.qkv(x).chunk(3, -1) # [B num_patches=h*w c]*3
        q, k, v = map(self.transpose_multi_head, qkv) 

        q = q*self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)

        out = paddle.matmul(attn, v) # [B num_heads num_patches head_dim]
        out = out.transpose([0, 2, 1, 3]) # [B num_patches num_heads head_dim]
        out = out.reshape([b, n, c])
        out = self.proj(out)
        return out

class SwinBlocks(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_norm = nn.LayerNorm(self.dim)
        self.attn = WindowAttention(self.dim, self.window_size, self.num_heads)
        self.mlp_norm = nn.LayerNorm(self.dim)
        self.mlp = MLP(self.dim)

    def forward(self, x):
        h, w = self.resolution
        b, n, c = x.shape

        clone = x
        x = self.attn_norm(x)
        x = x.reshape([b, h, w, c])
        x = windows_partition(x, self.window_size) # [b h w c] -> [b*num_patches//window_size//window_size window_size window_size c]
        x = x.reshape([-1, self.window_size*self.window_size, c]) # 将所有的小格全部拉出来，准备送入attention
        x = self.attn(x)
        x = x.reshape([-1, self.window_size, self.window_size, c])
        x = windows_reverse(x, self.window_size, h, w) # [b h w c]
        x = x.reshape([b, h*w, c])
        x += clone 

        clone = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x += clone
        return x

def main():
    t = paddle.randn([4, 3, 224, 224])
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    swin_block = SwinBlocks(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7)
    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)

    out = patch_embedding(t)
    print('patch_embedding:', out.shape)
    out = swin_block(out)
    print('swin_block:', out.shape)
    out = patch_merging(out)
    print('patch_merging:', out.shape)

if __name__ == '__main__':
    main()