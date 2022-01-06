import paddle
import paddle.nn as nn

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
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)

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
        attn_weight = attn
        # [B num_heads patch_nums patch_nums] * [B num_heads patch_nums head_dim] -> [B num_heads patch_nums head_dim]
        out = paddle.matmul(attn, v)
        # [B num_heads patch_nums head_dim] -> [B patch_nums num_heads head_dim]
        out = out.transpose([0, 2, 1, 3])

        # [B patch_nums num_heads head_dim] -> # [B patch_nums all_head_dim]
        out = out.reshape([B, N, -1])
        # [B patch_nums all_head_dim] -> [B patch_nums embed_dim]
        out = self.proj(out)
        return out, attn_weight

def main():
    t = paddle.randn([8, 16, 96])
    model = Attention(embed_dim=96, num_heads=4, qkv_bias=False, qk_scale=None)
    print(model)
    out, w = model(t)
    print(out.shape)
    print(w.shape)

if __name__ == "__main__":
    main()