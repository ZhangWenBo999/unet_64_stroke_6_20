from torch import nn
import torch
import math
# from model.swish import Swish
from torch.nn import functional as F
# from model.base_function import init_net
from einops import rearrange as rearrange
import numbers

class Downsample(nn.Module):
    def __init__(self, num_ch=32):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch*2, track_running_stats=False),
            nn.GELU()
        )

        #self.body = nn.Conv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, num_ch=32):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch//2, track_running_stats=False),
            nn.GELU()
        )

        #self.body = nn.Conv2d(in_channels=num_ch, out_channels=num_ch//2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.body(x)

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention_C_M(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, LayerNorm_type='WithBias'):
        super(Attention_C_M, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.GELU()
        )

    def forward(self, x_input):

        B, N, C = x_input.shape

        h = w = int(N**0.5)
        x = to_4d(x_input, h, w)

        b, c, h, w = x.shape
        x_1 = self.norm1(x)
        g = self.gate(x_1)

        qkv = self.qkv_dwconv(self.qkv(x_1))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        #attn = attn.softmax(dim=-1)
        attn = F.relu(attn)
        #attn = torch.square(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out * g
        out = self.project_out(out)
        #out = x+ out

        out = to_3d(out)

        return out

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

if __name__ == "__main__":
    # input = torch.randn(1, 32, 128, 128)
    input = torch.randn(1, 1024, 1024)
    model = Attention_C_M(dim=1024)
    output = model(input)
    print("input:", input.shape)
    print("output:", output.shape)

