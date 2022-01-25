from .hpb import HPB
from .et import EfficientTransformer
from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return ESRT(args)

# TODO: 3HPB feature concatenate to ET input
# LayerNorm
# pixel shuffle
class ESRT(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()

        self.args = args

        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]

        mlp_dim = args.mlp_dim

        lam_res = torch.nn.Parameter(torch.ones(1))
        lam_x = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lam_res, lam_x)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = conv(args.n_colors, n_feats, kernel_size)

        self.path1 = nn.Sequential(
            common.BackBoneBlock(3, HPB, inChannel=n_feats,
                                 outChannel=n_feats, reScale=self.adaptiveWeight),
            common.BackBoneBlock(1, EfficientTransformer,
                                 mlpDim=mlp_dim, inChannels=n_feats),
            conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        )

        self.path2 = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        )

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.head(x)
        x1, x2 = self.path1(x), self.path2(x)
        x = x1 + x2

        x = self.add_mean(x)
        
        return x
