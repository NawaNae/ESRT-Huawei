from .common import default_conv

import torch
import torch.nn as nn
import torch.nn.functional as F

# High Filter Module
class HFM(nn.Module):
    def __init__(self, k=2):
        super().__init__()

        self.k = k

        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.k, stride=self.k),
            nn.Upsample(scale_factor=self.k, mode='nearest'),
        )

    def forward(self, tL):
        assert tL.shape[2] % self.k == 0, 'h, w must divisible by k'
        return tL - self.net(tL)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ARFB: RU
class ResidualUnit(nn.Module):
    def __init__(self, inChannel, outChannel, reScale, kernelSize=1, bias=True):
        super().__init__()

        self.reduction = default_conv(
            inChannel, outChannel//2, kernelSize, bias)
        self.expansion = default_conv(
            outChannel//2, inChannel, kernelSize, bias)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        res = self.reduction(x)
        res = self.lamRes * self.expansion(res)
        x = self.lamX * x + res

        return x

class ARFB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()
        self.RU1 = ResidualUnit(inChannel, outChannel, reScale)
        self.RU2 = ResidualUnit(inChannel, outChannel, reScale)
        self.conv1 = default_conv(2*inChannel, 2*outChannel, kernel_size=1)
        self.conv3 = default_conv(2*inChannel, outChannel, kernel_size=3)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        x_ru1 = self.RU1(x)
        x_ru2 = self.RU2(x_ru1)
        x_ru = torch.cat((x_ru1, x_ru2), 1)
        x_ru = self.conv1(x_ru)
        x_ru = self.conv3(x_ru)
        x_ru = self.lamRes * x_ru
        x = x*self.lamX + x_ru
        return x

# CNN backbone main block
class HPB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()
        self.hfm = HFM()
        self.arfb1 = ARFB(inChannel, outChannel, reScale)
        self.arfb2 = ARFB(inChannel, outChannel, reScale)
        self.arfb3 = ARFB(inChannel, outChannel, reScale)
        self.arfbShare = ARFB(inChannel, outChannel, reScale)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.se = SELayer(inChannel)
        self.conv1 = default_conv(2*inChannel, outChannel, kernel_size=1)

    def forward(self, x):
        ori = x
        x = self.arfb1(x)
        x = self.hfm(x)
        x = self.arfb2(x)
        x_share = F.interpolate(x, scale_factor=0.5)
        for _ in range(5):
            x_share = self.arfbShare(x_share)
        x_share = self.upsample(x_share)

        x = torch.cat((x_share, x), 1)
        x = self.conv1(x)
        x = self.se(x)
        x = self.arfb3(x)
        x = ori+x
        return x