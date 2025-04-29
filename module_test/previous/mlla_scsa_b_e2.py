import sys
sys.path.append(r'D:\code\1130\256_fid_lpips_linux\o256_m12_fid_lpips')
from module_test.previous.SCSA_e2 import *
from module_test.previous.mlla_attnres_e2 import *


# 定义CBAM模块
class MLLA_SCSA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(MLLA_SCSA, self).__init__()

        self.scsa = SCSA(dim=in_channels,  head_num=8)
        self.MLLA = MLLAttention(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 并联
        scsa = self.scsa(x)
        mlla = self.MLLA(x)

        out = scsa * self.sigmoid(mlla)


        # # 串联
        # scsa = self.scsa(x)
        # mlla = self.MLLA(scsa)
        #
        # out = scsa * nn.sigmoid(mlla)

        return out
