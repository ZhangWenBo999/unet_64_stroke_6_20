
import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2d_BN(nn.Module):
    def __init__(self, in_features, out_features=None, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, dim, scales=(3, 5, 7), reduction_ratio=4):
        super(MultiScaleFeatureFusion, self).__init__()
        self.dim = dim
        self.scales = scales

        self.scale_convs = nn.ModuleList([
            Conv2d_BN(dim, dim // reduction_ratio, kernel_size=scale, padding=scale // 2)
            for scale in scales
        ])
        # fuse_conv 的输入通道数为多尺度分支拼接后的通道数
        self.fuse_conv = Conv2d_BN(dim // reduction_ratio * len(scales), dim, kernel_size=1)

    def forward(self, x):
        # 分别通过不同尺度的卷积分支
        scale_features = [conv(x) for conv in self.scale_convs]
        # 通道拼接多尺度特征
        fused_features = torch.cat(scale_features, dim=1)  # 拼接后通道数为 dim // reduction_ratio * len(scales)
        # 融合后的输出
        return self.fuse_conv(fused_features)

if __name__ == "__main__":
    input = torch.randn(1, 32, 128, 128)
    model = MultiScaleFeatureFusion(dim=32)
    output = model(input)
    print("input:", input.shape)
    print("output:", output.shape)
