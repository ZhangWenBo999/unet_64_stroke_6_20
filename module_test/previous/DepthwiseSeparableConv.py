
import torch
import torch.nn as nn

# class DepthwiseSeparableConv(nn.Module):
#     """使用深度可分离卷积减少计算复杂度"""
#     def __init__(self, inp, outp, kernel_size, stride=1, padding=0):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp)
#         self.pointwise = nn.Conv2d(inp, outp, kernel_size=1, stride=1)
#
#     def forward(self, x):
#         return self.pointwise(self.depthwise(x))


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=kernel_size // 2, groups=in_channels, bias=False)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWConv3_5_cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWConv3_5_cat, self).__init__()
        self.dwconv3 = DepthwiseSeparableConv2d(in_channels, out_channels, 3)
        self.dwconv5 = DepthwiseSeparableConv2d(in_channels, out_channels, 5)
        self.fuse_dwconv = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x):
        return self.fuse_dwconv(torch.cat([self.dwconv3(x), self.dwconv5(x)], dim=1))


if __name__ == "__main__":
    input = torch.randn(1, 32, 128, 128)
    model = DWConv3_5_cat(32, 32)
    output = model(input)
    print("input:", input.shape)
    print("output:", output.shape)
