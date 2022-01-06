import paddle
import paddle.nn as nn

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

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

    def forward(self, x, mask=None):
        b, n, c = x.shape # [B*num_windows ws=window_size ws c] num_patches=ws*ws
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multi_head, qkv) 

        q = q*self.scale
        attn = paddle.matmul(q, k, transpose_y=True) # [B*num_windows num_heads num_patches num_patches]

        # relative position bias
        # TODO
        # mask
        if mask is None:
            attn = self.softmax(attn)
        else:
            # mask: [num_windows num_patches num_patches]
            # attn: [B*num_windows num_heads num_patches num_patches] -> [B num_windows num_heads num_patches num_patches]
            attn = attn.reshape([b//mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1]])
            # mask: [1 num_windows 1         num_patches num_patches]
            # attn: [B num_windows num_heads num_patches num_patches]
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, mask.shape[1], mask.shape[1]])

        out = paddle.matmul(attn, v) 
        out = out.transpose([0, 2, 1, 3]) # [B num_patches num_heads head_dim]
        out = out.reshape([b, n, c])
        out = self.proj(out)
        return out

def generate_mask(window_size=4, shift_size=2, input_resolution=(8, 8)):
    h, w = input_resolution
    window_mask = paddle.zeros([1, h, w, 1])
    h_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)] # a[slice(0, x)] = a[0:x]
    w_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]
    
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            window_mask[:, h, w, :] = cnt
            cnt += 1
    masked_windows = windows_partition(window_mask, window_size)
    masked_windows = masked_windows.reshape([-1, window_size*window_size])
    # [n, 1, window_size*window_size] 相同的n可以不管，假装没有这个维度；也就是从一维window_size*window_size，变成行二维和列二维
    # [n, window_size*window_size, 1]
    attn_mask = masked_windows.unsqueeze(1) - masked_windows.unsqueeze(2) # 广播
    attn_mask = paddle.where(attn_mask != 0,
                             paddle.ones_like(attn_mask)*255,
                             paddle.zeros_like(attn_mask))
    return attn_mask

class SwinBlock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.resolution)
        self.attn_norm = nn.LayerNorm(self.dim)
        self.attn = WindowAttention(self.dim, self.window_size, self.num_heads)
        self.mlp_norm = nn.LayerNorm(self.dim)
        self.mlp = MLP(self.dim)
        if self.shift_size > 0:
            attn_mask = generate_mask(self.window_size, self.shift_size, self.resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask) # 将变量注册为一个层

    def forward(self, x):
        h, w = self.resolution
        b, n, c = x.shape

        clone = x
        x = self.attn_norm(x)
        x = x.reshape([b, h, w, c])

        # shift window
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x
        # compute window attention
        x = windows_partition(shifted_x, self.window_size) # [b h w c] -> [b*num_patches//window_size//window_size window_size window_size c]
        x = x.reshape([-1, self.window_size*self.window_size, c]) # 将所有的小格全部拉出来，准备送入attention
        x = self.attn(x, mask=self.attn_mask)
        x = x.reshape([-1, self.window_size, self.window_size, c])
        shifted_x = windows_reverse(x, self.window_size, h, w) # [b h w c]
        # shift back
        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
     
        x = x.reshape([b, h*w, c])
        x += clone 

        clone = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x += clone
        return x

class SwinStage(nn.Layer):
    def __init__(self, dim, input_resolution, depths, num_heads, window_size, patch_merging=None):
        super().__init__()
        self.blocks = nn.LayerList()
        for i in range(depths):
            self.blocks.append(SwinBlock(dim=dim,
                                         input_resolution=input_resolution,
                                         num_heads= num_heads,
                                         window_size= window_size,
                                         shift_size=0 if (i%2==0) else window_size//2))
        if patch_merging is None:
            self.patch_merging = Identity()
        else:
            self.patch_merging = patch_merging(input_resolution, dim) # patch_merging只是传入了一个class名字

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.patch_merging(x)
        return x

class Swin(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96,
                 window_size=7,
                 num_heads=[3, 6, 12, 24],
                 depths=[2, 2, 6, 2],
                 num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_stages = len(depths)
        self.num_features = int(self.embed_dim * 2**(self.num_stages-1))
        self.patch_resolution = [image_size // patch_size, image_size // patch_size]
        self.patch_embedding = PatchEmbedding(patch_size, self.embed_dim)
        self.stages = nn.LayerList()
        for idx, (depths, num_heads) in enumerate(zip(self.depths, self.num_heads)):
            stage = SwinStage(dim=int(self.embed_dim * 2**idx),
                              input_resolution=(self.patch_resolution[0]//(2**idx),
                                                self.patch_resolution[0]//(2**idx)),
                              depths=depths,
                              num_heads= num_heads,
                              window_size= window_size,
                              patch_merging=PatchMerging if (idx < self.num_stages-1) else None)
            self.stages.append(stage)
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1D(1)
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        for stage in self.stages:
            x  = stage(x)
        x = self.norm(x)
        x = x.transpose([0, 2, 1]) # [b num_features num_windows]
        x = self.avgpool(x) # [b num_features 1]
        x = x.flatten(1) # [b num_features]
        x = self.fc(x)
        return x

def main():
    # 生成mask，并可视化
    # mask = generate_mask()
    # print(mask.shape)
    # mask = mask.cpu().numpy().astype('uint8')
    # for i in range(4): # 4个16 x 16window 
    #     for j in range(16):
    #         for k in range(16):
    #             print(mask[i, j, k], end='\t')
    #             print()
    #         img = Image.fromarray(mask[i, :, :])
    #         img.save(f'{i}.png')
    #         print()
    #         print()
    #     print()

    # 生成单个swin步骤
    # patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    # swin_block_w_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=0)
    # swin_block_sw_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=7//2)
    # patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)
    # t = paddle.randn([4, 3, 224, 224])
    # print('Image:', t.shape)
    # out = patch_embedding(t)
    # print('patch_embedding out:', out.shape)
    # out = swin_block_w_msa(out)
    # out = swin_block_sw_msa(out)
    # print('swin_block out:', out.shape)
    # out = patch_merging(out)
    # print('patch_merging out:', out.shape)

    t = paddle.randn([4, 3, 224, 224])
    model = Swin()
    print(model)
    out = model(t)
    print(out.shape)

if __name__ == '__main__':
    main()