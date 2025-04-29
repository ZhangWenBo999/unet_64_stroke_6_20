import logging
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class CSwinAttention_block(nn.Module):
    def __init__(self, dim, split_size=4, dim_out=None, num_heads=2, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.cswinln = nn.LayerNorm(dim, eps=1e-4)

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.sp = split_size

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        # lepe conv from set_lepe_conv

        # self.proj = nn.LazyConv2d(dim, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def set_lepe_conv(self, x: torch.Tensor) -> nn.Conv2d:
        """input_size : [B, C, H', W']"""
        dim = x.size(1)
        # init lepe conv
        # self.lepe_conv = nn.LazyConv2d(dim, kernel_size=3, stride=1, padding=1, groups=dim).to(x.device)
        self.lepe_conv = nn.Conv2d(dim,dim, kernel_size=3, stride=1, padding=1, groups=dim).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input_size : [B, C, H, W]  or [B, L, C]"""

        # create a list for later concat
        attened_x = []
        attened_att = []

        if len(x.shape) == 3:  # [B, L, C]
            B, L, C = x.shape
            H = W = int(np.sqrt(L))

        elif len(x.shape) == 4:  # [B, C, H, W]
            B, C, H, W = x.shape

        assert H % self.sp == 0 and W % self.sp == 0,\
            f'H={H} or W={W} cannot be divided by split_size={self.sp} '

        condition = (H == self.sp and W == self.sp)  # feature size == split size, one attn operation
                                                     # feature size  > split size, two attn operations

        if condition:
            h = w = 1
            hsp = wsp = self.sp  # full feature
            param = [(h, hsp, w, wsp)]

        else:
            h1, hsp_1, w_1, wsp_1 = 1, H, W // self.sp, self.sp  # vertical
            h2, hsp_2, w_2, wsp_2 = H // self.sp, self.sp, 1, W  # horizontal
            param = [(h1, hsp_1, w_1, wsp_1), (h2, hsp_2, w_2, wsp_2)]

        if len(x.shape) == 3:  # already in patch-form [B, (H*W), C]
            x_patch = x
        if len(x.shape) == 4:  # [B, C, H, W] to [B, (H*W), C]
            x_patch = rearrange(x, 'b c h w -> b (h w) c')

        x_patch = self.cswinln(x_patch)
        qkv = self.to_qkv(x_patch).chunk(3, dim=-1)


        if condition:
           qkv = [qkv]

        else:
            # split channel [:C // 2] , [C // 2:] (h w) = l
            qkv = map(lambda t: rearrange(t, 'b l (split c)  -> split b l c', split=2), qkv)
            (q1, q2), (k1, k2), (v1, v2) = qkv
            qkv = [(q1, k1, v1), (q2, k2, v2)]

        for index, (x, (h, hsp, w, wsp)) in enumerate(zip(qkv, param)):

            # print(h, hsp, w, wsp)
            # cswin format
            q, k, v = map(lambda t: rearrange(t, 'b (h hsp w wsp) (c head)  -> (b h w) head (hsp wsp) c',
                                              head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp), x)

            # print(f'{q.shape=},{k.shape=}, {v.shape=} ')
            """
            from
            [B, (H*W)        , C       ]
            [b, (h hsp w wsp), (c head)]
            to
            [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head]
            [(b h w)            , head, (hsp wsp)  , c     ]
            Note:
            c = C / self.num_heads, head = self.mun_heads
            h = H / self.sp      , hsp  = self.sp
            w = W / self.sp      , wsp  = self.sp
            """

            # from [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head] to [(B * H/hsp * W/wsp), C, hsp, wsp]
            lepe = rearrange(v, '(b h w) head (hsp wsp) c -> (b h w) (c head) hsp wsp',
                             head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

            # set lepe_conv
            self.set_lepe_conv(lepe)  ###

            # lepe_conv
            lepe = self.lepe_conv(lepe)

            # back to [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head]
            lepe = rearrange(lepe, '(b h w) (c head) hsp wsp -> (b h w) head (hsp wsp) c',
                             head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)
            # print(f'{lepe.shape=}')

            # attention
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
            attn = self.attn_drop(attn)

            x = (attn @ v) + lepe

            # [(B * H / hsp * W / wsp), head, (hsp * wsp), C / head] to[(B , C, H, W]
            x = rearrange(x, '(b h w) head (hsp wsp) c -> b (c head) (h hsp) (w wsp)',
                             head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

            attened_x.append(x)
            attened_att.append(attn)

        x = self.proj(torch.cat(attened_x, dim=1))


        # return x, attened_att

        # print("input:", x.shape)
        # print("output:", attened_att.shape)

        # return x

        return to_3d(x)



if __name__ == "__main__":
    # input = torch.randn(1, 32, 128, 128)
    input = torch.randn(1, 1024, 1024)
    model = CSwinAttention_block(dim=1024)
    output = model(input)
    print("input:", input.shape)
    print("output:", output.shape)

