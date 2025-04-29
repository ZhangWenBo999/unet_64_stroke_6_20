"""
二、多层次特征融合
1. 缺陷：目前SCSA模块主要基于单一层级的空间和通道注意力信息

2. 解决方案：引入多层次特征融合，例如在不同网络层级上对特征图进行聚合，或者在不同分辨率的特征图之间实现交互。
通过融合低层次的细节信息和高层次的语义信息，增强模型对复杂结构和细节的捕捉能力。进一步提升图像分割、目标检测等
任务中细节处理和物体边界识别的效果。提高模块对多语义空间信息的利用效率，使得特征表达更加丰富。
"""
import typing as t
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """

    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def generate_rotations(self, x):
        # 获取输入张量的形状
        *channel_dims, feature_dim = x.shape[1:-1][0], x.shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0, "Feature dimension must be divisible by 2 * k_max"
        # 生成角度
        theta_ks = 1 / (self.base ** (torch.arange(k_max, dtype=x.dtype, device=x.device) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d, dtype=x.dtype, device=x.device) for d in channel_dims],
                                           )], dim=-1)
        # 计算旋转矩阵的实部和虚部
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        return rotations

    def forward(self, x):
        # 生成旋转矩阵
        rotations = self.generate_rotations(x)
        # 将 x 转换为复数形式
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        # 应用旋转矩阵
        pe_x = torch.view_as_complex(rotations) * x_complex
        # 将结果转换回实数形式并展平最后两个维度
        return torch.view_as_real(pe_x).flatten(-2)


class MultiLevelSCSA_scaler(nn.Module):
    def __init__(self, dim, levels=4, group_kernel_sizes=[3, 5, 7, 9], qkv_bias=False, attn_drop_ratio=0.):
        super(MultiLevelSCSA_scaler, self).__init__()
        self.dim = dim
        self.scaler = self.dim ** -0.5
        # self.head_num = head_num
        # self.head_dim = dim // head_num
        self.levels = levels  # 表示要融合的特征层级数
        # self.group_chans = dim // 4

        # 每层级独立的卷积和自注意力模块
        self.multi_layer_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)
            for k in group_kernel_sizes
        ])
        self.multi_layer_norms = nn.ModuleList([nn.GroupNorm(4, dim) for _ in range(self.levels)])
        self.sa_gate = nn.Sigmoid()

        self.fusion_weights = nn.Parameter(torch.ones(levels))  # 可学习的融合权重
        # 初始化融合权重
        nn.init.constant_(self.fusion_weights, 1/levels)

        # 融合后的通道注意力
        # self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        # self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        # self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop_ratio)

        # self.dim = dim
        # self.input_resolution = input_resolution
        self.num_heads = 4
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()

    def forward(self, x):
        layer_outputs = []
        for conv, norm in zip(self.multi_layer_convs, self.multi_layer_norms):
            layer_out = self.sa_gate(norm(conv(x)))
            layer_outputs.append(layer_out)

        # # 多层级融合特征
        # fused_features = sum(layer_outputs) / self.levels

        # 加权特征融合
        weights = F.softmax(self.fusion_weights, 0)
        fused_features = sum(w * f for w, f in zip(weights, layer_outputs))

        x = fused_features

        x = x.permute(0,2,3,1).reshape((x.size(0), x.size(2) * x.size(3), x.size(1)))
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        # self.rope = RoPE(shape=(h, w, self.dim))
        num_heads = self.num_heads
        head_dim = c // num_heads

        x1 = x.reshape(-1, x.shape[-1])
        # qk = self.kan(x1).reshape(b, n, 2 * c)
        qk = self.qk(x1).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        qk = qk.reshape(b, n, 2, c).permute(2, 0, 1, 3)

        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        x = x.transpose(2, 1).reshape((b, c, h, w))
        return x


        # # 通道注意力
        # y = self.q(fused_features) @ self.k(fused_features).transpose(-2, -1) * self.scaler
        # y = self.attn_drop(y.softmax(dim=-1))

        # return fused_features * y

if __name__ == '__main__':
    input = torch.randn(1, 128, 64, 64)
    model = MultiLevelSCSA_scaler(dim=128)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
