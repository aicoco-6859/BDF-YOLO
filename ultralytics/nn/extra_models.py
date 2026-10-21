import math

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import init
from timm.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from torch.nn import init, Parameter
from .rmt import RetBlock, RelPos2d
from torchvision.ops import DeformConv2d

__all__ = (
    "PixelAttention_CGA",
    "CAFM",
    "CAFMFusion",
    "SpatialAttention_CGA",
    "ChannelAttention_CGA",
    "CGAFusion",
    "Add_Conv",
    "ContextGuideFusionModule",
    "ContextGuideFusionModule_SE",
    "ContextGuideFusionModule_CBAM",
    "SEAttention",
    "CSP_PTB",
    "C2f_PPA",
    "FeaturePyramidSharedConv",
    "EIEStem",
    "C2f_EIEM",
    "SPDConv",
    "CSPOmniKernel",
    "PyramidContextExtraction",
    "GetIndexOutput",
    "FuseBlockMulti",
    "DynamicInterpolationFusion",
    "RCM",
    "Add_F",
    "Concat3",
    "Concat_C",
    "Concat_S",
    "Concat_CS",
    "Concat_1",
    "ResidualExciteFusion_rgb",
    "ResidualExciteFusion_depth",
    "ResidualExciteFusion_1",
    "PAdd",
    "PAdd_Conv",
    "C2f_RetBlock",
    "Concat_F",
    "FocusFeature",
    "AxialFusionUnit",
    "ContextGuideFusionModule_DCSAM",
    "LA",
    "DAA",
    "GC",
    "LDCFM",
    "LDCFM_D",
    "ContextGuideFusionModule_noDCSAM",
    "ContextGuideFusionModule_MGDAM",
    "CMGA",
    "n_DCFM",
    "CGFM_1",
    "CGFM_12",
    "CGFM_13",
    "CGFM_132",
    "Add_2",
    "Add_3",
    "Add_23",
    "Add_32",
    "CGFM_1_2_3",
    "Add_1_2_3",
    "LA_1M",
    "LA_2M",
    "LA_12",
    "LA_M",
    "LA_1_2_M",
    "LA_2_1_M"

)


######=============================Add改进 begin=============================######
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2],  # 每隔一个像素取值
                                    x[..., 1::2, ::2],  # 从第二个像素开始，每隔一个像素取值
                                    x[..., ::2, 1::2],  # 每隔一个像素取值，从第二列开始
                                    x[..., 1::2, 1::2]),  # 从第二个像素和第二列开始，每隔一个像素取值
                                   1))  # 在通道纬度上拼接
        # return self.conv(self.contract(x))


class Add(nn.Module):
    #  Add two tensors
    # 输入张量和输出张量的形状完全相同，都是 [N, C, H, W]
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        # 加法
        return torch.add(x[0], x[1])
        # 乘法
        # return torch.mul(x[0], x[1])


# parameter与add结合
class PAdd_Conv(nn.Module):
    #  Add two tensors
    def __init__(self, c1):
        super(PAdd_Conv, self).__init__()
        self.rgb = Parameter(torch.ones(1))
        self.depth = Parameter(torch.ones(1))
        self.conv = nn.Conv2d(c1, c1, 1, 1)

    def forward(self, x):
        # 加法
        x = torch.add(self.rgb * x[0], self.depth * x[1])
        return self.conv(x)
        # 乘法
        # return torch.mul(x[0], x[1])


class PAdd(nn.Module):
    #  Add two tensors
    def __init__(self, c1):
        super(PAdd, self).__init__()
        self.rgb = Parameter(torch.ones(1))
        self.depth = Parameter(torch.ones(1))
        self.conv = nn.Conv2d(c1, c1, 1, 1)

    def forward(self, x):
        # 加法
        return torch.add(self.rgb * x[0], self.depth * x[1])
        # 乘法
        # return torch.mul(x[0], x[1])


class Add_Conv(nn.Module):
    #  Add two tensors
    def __init__(self, c1):
        super(Add_Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c1, 1, 1)

    def forward(self, x):
        # 加法
        x = torch.add(x[0], x[1])
        return self.conv(x)
        # 乘法
        # return torch.mul(x[0], x[1])


# class Add_F_1(nn.Module):
#     def __init__(self, arg):
#         super(Add_F_1, self).__init__()
#         self.arg = arg
#         self.conv = nn.Conv2d(512, 512, 3, 2, autopad(3, None, 1), groups=1, dilation=1, bias=False)
#
#     def forward(self, x):
#         return self.conv(torch.add(x[0], x[1]))


class Add_F(nn.Module):
    def __init__(self, c1, c2, DC=True):
        super().__init__()
        self.DC = DC
        if self.DC:
            self.conv = Conv(c2, c1, 3, 2)
            # self.conv = nn.Conv2d(c2, c1, 3, 2, autopad(3, None, 1), groups=1, dilation=1, bias=False)
            self.conv1 = nn.Conv2d(c1 * 2, c1, 1, 1)
        else:
            self.conv1 = nn.Conv2d(c1, c1, 1, 1)

    def forward(self, x):  # 0 1 (2)
        fusion = x[0] + x[1]
        if not self.DC:
            return self.conv1(fusion)
        else:
            return self.conv1(torch.cat([fusion, self.conv(x[2])], 1))


# class Add_F_3(nn.Module):
#     def __init__(self, arg):
#         super(Add_F_3, self).__init__()
#         self.arg = arg
#         self.conv = nn.Conv2d(512, 512, 3, 2, autopad(3, None, 1), groups=1, dilation=1, bias=False)
#
#     def forward(self, x):
#         sumadd = torch.add(x[0], x[1])
#         return self.conv(torch.add(sumadd, x[2]))


######=============================Add改进 end=============================######


######=============================Concat3改进 begin=============================######

class Concat3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, ratio=16, kernel_size=7, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.spatial_attention = SpatialAttention_Concat(7)
        self.fin_channel_attention = ChannelAttention_Concat(c1, ratio)
        self.conv = nn.Conv2d(c1 * 2, c1, 1, 1)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        weight1 = self.spatial_attention(x1)
        weight2 = self.spatial_attention(x2)
        weight = (weight1 / weight2)
        x2 = weight * x2
        x1 = x1 * (2 - weight)
        x = torch.cat((x1, x2), self.d)
        x = self.conv(x)
        x = x * self.fin_channel_attention(x)
        return x


class ChannelAttention_Concat(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_Concat, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # 写法二,亦可使用顺序容器
        # self.sharedMLP = nn.Sequential(
        # nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
        # nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention_Concat(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_Concat, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x1 = torch.mean(x)
        x2 = torch.max(x)
        x = x1 + x2
        x = self.sigmoid(x)
        return x


class Concat_C(nn.Module):
    """
    双流各自进行通道注意力，然后Concat，然后1*1的conv，然后再通道注意力
    """

    def __init__(self, c1, ratio=16, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.channel_attention1 = ChannelAttention_Concat(c1, ratio)
        self.channel_attention2 = ChannelAttention_Concat(c1, ratio)
        self.conv1x1 = nn.Conv2d(c1 * 2, c1, 1, 1)
        self.final_channel_attention = ChannelAttention_Concat(c1, ratio)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        x1 = x1 * self.channel_attention1(x1)
        x2 = x2 * self.channel_attention2(x2)

        x = torch.cat((x1, x2), self.d)
        x = self.conv1x1(x)
        x = x * self.final_channel_attention(x)

        return x


class Concat_S(nn.Module):
    """双流各自进行空间注意力，然后Concat，然后1*1的conv，然后再空间注意力"""

    def __init__(self, c1, ratio=16, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.spatialattention1 = SpatialAttention_Concat(7)
        self.spatialattention2 = SpatialAttention_Concat(7)
        self.conv1x1 = nn.Conv2d(c1 * 2, c1, 1, 1)
        self.final_spatial_attention = SpatialAttention_Concat(7)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        x1 = x1 * self.spatialattention1(x1)
        x2 = x2 * self.spatialattention2(x2)

        x = torch.cat((x1, x2), self.d)
        x = self.conv1x1(x)
        x = x * self.final_spatial_attention(x)

        return x


class Concat_CS(nn.Module):
    """双流各自进行通道注意力，然后各自空间注意力，然后Concat，然后1*1的conv，然后再空间注意力"""

    def __init__(self, c1, ratio=16, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.spatialattention1 = SpatialAttention_Concat(7)
        self.spatialattention2 = SpatialAttention_Concat(7)
        self.channelattention1 = ChannelAttention_Concat(c1, ratio)
        self.channelattention2 = ChannelAttention_Concat(c1, ratio)
        self.conv1x1 = nn.Conv2d(c1 * 2, c1, 1, 1)
        self.final_spatial_attention = SpatialAttention_Concat(7)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        x1_att = x1 * self.channelattention1(x1)
        x1_att = x1_att * self.spatialattention1(x1_att)

        x2_att = x2 * self.channelattention2(x2)
        x2_att = x2_att * self.spatialattention2(x2_att)

        x = torch.cat((x1_att, x2_att), self.d)
        x = self.conv1x1(x)
        x = x * self.final_spatial_attention(x)

        return x


class Concat_1(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, ratio=16, kernel_size=7, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.channelattention1 = ChannelAttention_Concat(c1, ratio)
        self.channelattention2 = ChannelAttention_Concat(c1, ratio)
        self.spatial_attention = SpatialAttention_Concat(7)
        self.fin_channel_attention = ChannelAttention_Concat(c1, ratio)
        self.conv = nn.Conv2d(c1 * 2, c1, 1, 1)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        x1_att = x1 * self.channelattention1(x1)
        x2_att = x2 * self.channelattention1(x2)

        weight1 = self.spatial_attention(x1_att)
        weight2 = self.spatial_attention(x2_att)
        weight = (weight1 / weight2)
        x2 = weight * x2
        x1 = x1 * (2 - weight)

        x = torch.cat((x1, x2), self.d)
        x = self.conv(x)
        x = x * self.fin_channel_attention(x)
        return x


class Concat_F(nn.Module):
    def __init__(self, c1, c2, ratio=16, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.spatial_attention = SpatialAttention_Concat(7)
        self.fin_channel_attentiopn = ChannelAttention_Concat(c1, ratio)
        self.conv1 = nn.Conv2d(c1 * 2, c1, 1, 1)
        self.conv = Conv(c2, c1, 3, 2)

    def forward(self, x):  # 0 1 (2)
        x1, x2, x3 = x
        x3 = self.conv(x3)
        weight1 = self.spatial_attention(x1)
        weight2 = self.spatial_attention(x2)
        weight3 = self.spatial_attention(x3)

        total_weight = weight1 + weight2 + weight3
        weight1 = weight1 / total_weight
        weight2 = weight2 / total_weight
        weight3 = weight3 / total_weight

        x1 = x1 * weight1
        x2 = x2 * weight2
        x3 = x3 * weight3
        x = torch.cat((x1, x2, x3), self.d)
        x = self.conv1(x)
        x = x * self.fin_channel_attentiopn(x)
        return x


class CM(nn.Module):
    def __init__(self, dim, ratio=16) -> None:
        super().__init__()

        self.dim = dim
        self.d = 1  # 沿着哪个维度进行拼接
        self.spatial_attention = SpatialAttention_Concat(7)
        self.fin_channel_attention = ChannelAttention_Concat(48, ratio)
        self.conv = nn.Conv2d(96, 48, 1, 1)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        weight1 = self.spatial_attention(x1)
        weight2 = self.spatial_attention(x2)
        weight = (weight1 / weight2)
        x2 = weight * x2
        x1 = x1 * (2 - weight)
        x = torch.cat((x1, x2), self.d)
        x = self.conv(x)
        x = x * self.fin_channel_attention(x)
        return torch.split(x, self.dim, dim=1)


######=============================Concat3改进 end=============================######

######=============================REF改进 begin=============================######
class Excitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(Excitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = self.fc(x)
        y = x * weighting
        return y


class ResidualExciteFusion(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ResidualExciteFusion, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                 activation=activation)
        self.se_depth = Excitation(channels_in,
                                   activation=activation)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        if rgb.sum().item() < 1e-6:
            out = rgb + depth
        else:
            rgb_se = self.se_rgb(rgb)
            out = rgb + rgb_se + depth

        return out


class ResidualExciteFusion_1(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ResidualExciteFusion_1, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                 activation=activation)
        self.se_depth = Excitation(channels_in,
                                   activation=activation)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        depth_se = self.se_depth(depth)
        rgb_se = self.se_rgb(rgb)
        out_1 = rgb + rgb_se + depth_se
        out_2 = depth + rgb_se + depth_se

        return out_1, out_2


class ResidualExciteFusion_rgb(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ResidualExciteFusion_rgb, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                 activation=activation)
        self.se_depth = Excitation(channels_in,
                                   activation=activation)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        # if depth.sum().item() < 1e-6:
        #     pass
        # else:
        depth_se = self.se_depth(depth)

        if rgb.sum().item() < 1e-6:
            return rgb + depth
        else:
            rgb_se = self.se_rgb(rgb)
            out = torch.cat([rgb_se, depth_se], dim=0)
            out_rgb = torch.cat([out, rgb], dim=0)

        return out_rgb


class ResidualExciteFusion_depth(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ResidualExciteFusion_depth, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                 activation=activation)
        self.se_depth = Excitation(channels_in,
                                   activation=activation)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        # if depth.sum().item() < 1e-6:
        #     pass
        # else:
        depth_se = self.se_depth(depth)

        if rgb.sum().item() < 1e-6:
            return rgb + depth
        else:
            rgb_se = self.se_rgb(rgb)
            out = torch.cat([rgb_se, depth_se], dim=0)
            out_depth = torch.cat([out, depth], dim=0)

        return out_depth


######=============================REF end=============================######

######=============================CGAFusion begin=============================######
class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_CGA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        """
        dim:输入通道数
        """
        super(PixelAttention_CGA, self).__init__()
        # 一个深度可分离卷积层，输入通道数为 2 * dim，输出通道数为 dim，卷积核大小为 7x7，填充模式为反射填充（padding_mode='reflect'），偏置项开启（bias=True）。
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    # pattn1为第一层注意力图，形状同x
    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        # 将 x 和 pattn1 在新的维度上扩展，使得它们的形状变为 (B, C, 1, H, W)。
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        # 使用 torch.cat 函数沿着新扩展的维度（即索引为 2 的维度）拼接这两个张量，得到一个新的张量 x2，其形状为 (B, C, 2, H, W)。
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        # 使用 rearrange 函数重新排列 x2 的维度，使其形状变为 (B, 2*C, H, W)，以便能够输入到卷积层 pa2 中。
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        # 通过 pa2 卷积层计算得到第二层的注意力图 pattn2，然后通过 Sigmoid 激活函数将其转换为 [0, 1] 区间内的值
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, c1, c2, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(c2, reduction)
        self.pa = PixelAttention_CGA(c2)
        self.conv = nn.Conv2d(c2, c2, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.change = nn.Conv2d(c1, c2, 1, 1)

    def forward(self, data):
        x, y = data
        if x.shape[1] != y.shape[1]:
            x = self.change(x)
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


######=============================CGAFusion eng=============================######

######=============================CAFMFusion begin=============================######
# 像素级注意力

class CAFM(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output


class CAFMFusion(nn.Module):
    def __init__(self, c1, c2, heads):
        """
        c1:第一个层的通道数
        c2:第二个层的通道数
        heads:多头注意力机制中的头数
        """
        super(CAFMFusion, self).__init__()
        self.cfam = CAFM(c2, num_heads=heads)
        self.pa = PixelAttention_CGA(c2)
        self.conv = nn.Conv2d(c2, c2, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        # 输入通道数为c1，输出通道数为c2，用来将输出通道改为c2的大小
        self.change = nn.Conv2d(c1, c2, 1, 1)

    def forward(self, data):
        x, y = data
        # 如果x的输出通道数和y的输出通道数不一样，则将x的输出通道数改成和y一样
        if x.shape[1] != y.shape[1]:
            x = self.change(x)
        initial = x + y
        pattn1 = self.cfam(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


######=============================CAFMFusion end=============================######


######################################## ContextGuideFusionModule begin ########################################
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


# class SEAttention(nn.Module):
#     def __init__(self, channel=512, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


class CSAM(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()

        # 通道注意力分支（原SE模块）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力分支
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),  # 输入2通道：max和avg
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        # ----------------- 通道注意力 -----------------
        ca_weight = self.channel_attention(x)  # 形状: (B, C, 1, 1)
        x_ca = x * ca_weight  # 通道加权

        # ----------------- 空间注意力 -----------------
        # 沿通道维度计算max和avg，拼接为2通道
        max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]  # 形状: (B, 1, H, W)
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)  # 形状: (B, 1, H, W)
        spatial_feat = torch.cat([max_pool, avg_pool], dim=1)  # 形状: (B, 2, H, W)

        sa_weight = self.spatial_attention(spatial_feat)  # 形状: (B, 1, H, W)
        x_sa = x_ca * sa_weight  # 空间加权

        return x_sa


class ContextGuideFusionModule_CSAM(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.se = CSAM(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


# class DynamicSpatialAttention(nn.Module):
#     def __init__(self, kernel_sizes=[3, 5, 7], reduction=16):
#         super().__init__()
#         self.kernel_sizes = kernel_sizes
#         self.num_kernels = len(kernel_sizes)
#
#         # 多分支卷积（不同kernel size）
#         self.conv_branches = nn.ModuleList([
#             nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
#             for k in kernel_sizes
#         ])
#
#         # 动态权重生成器
#         self.weight_generator = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(2, self.num_kernels, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         spatial_feat = torch.cat([max_pool, avg_pool], dim=1)
#
#         # 生成各分支权重 (B×K×1×1)
#         branch_weights = self.weight_generator(spatial_feat)
#
#         # 多分支卷积 + 加权融合
#         sa_maps = []
#         for i, conv in enumerate(self.conv_branches):
#             sa = conv(spatial_feat)  # 各分支输出 (B×1×H×W)
#             weight = branch_weights[:, i:i + 1, :, :]  # 取对应分支权重
#             sa_weighted = sa * weight.expand_as(sa)
#             sa_maps.append(sa_weighted)
#
#         # 求和并Sigmoid
#         sa_fused = torch.sum(torch.stack(sa_maps, dim=0), dim=0)
#         sa_weight = self.sigmoid(sa_fused)
#         return sa_weight
#
#
# class DCSAM(nn.Module):
#     def __init__(self, channel=512, reduction=16, kernel_sizes=[3, 5, 7]):
#         super().__init__()
#         # 通道注意力（与原CSAM一致）
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         # 动态空间注意力
#         self.spatial_attention = DynamicSpatialAttention(kernel_sizes, reduction)
#
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # 通道注意力
#         ca_weight = self.channel_attention(x)
#         x_ca = x * ca_weight
#
#         # 动态空间注意力
#         sa_weight = self.spatial_attention(x_ca)
#         x_sa = x_ca * sa_weight
#
#         return x_sa
#
#
# class ContextGuideFusionModule_DCSAM(nn.Module):
#     def __init__(self, inc) -> None:
#         super().__init__()
#
#         self.adjust_conv = nn.Identity()
#         if inc[0] != inc[1]:
#             self.adjust_conv = Conv(inc[0], inc[1], k=1)
#
#         self.se = DCSAM(inc[1] * 2)
#
#     def forward(self, x):
#         x0, x1 = x
#         x0 = self.adjust_conv(x0)
#
#         x_concat = torch.cat([x0, x1], dim=1)  # n c h w
#         x_concat = self.se(x_concat)
#         x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
#         x0_weight = x0 * x0_weight
#         x1_weight = x1 * x1_weight
#         return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)

class SEAttention(nn.Module):
    def __init__(self, c2, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ContextGuideFusionModule(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.se = SEAttention(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


class ContextGuideFusionModule_SE(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.se = SEAttention(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return self.se(torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1))


class CBAM_ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class CBAM_SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = CBAM_ChannelAttention(c1)
        self.spatial_attention = CBAM_SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class ContextGuideFusionModule_CBAM(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.se = CBAM(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


######################################## ContextGuideFusionModule end ########################################

######################################## CSP-PTB(Partially Transformer Block) begin ########################################
class PSA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
                (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    # def forward(self, x):
    #     x, v = self.fc1(x).chunk(2, dim=1)
    #     x = self.dwconv(x) * v
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.drop(x)
    #     return x

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class MHSA_CGLU(nn.Module):
    def __init__(self,
                 inc,
                 drop_path=0.1,
                 ):
        super().__init__()
        self.norm1 = LayerNorm2d(inc)
        self.norm2 = LayerNorm2d(inc)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(inc)
        self.mhsa = PSA_Attention(inc, num_heads=8)

    def forward(self, x):
        shortcut = x
        x = self.drop_path(self.mhsa(self.norm1(x))) + shortcut
        x = self.drop_path(self.mlp(self.norm2(x))) + x
        return x


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class PartiallyTransformerBlock(nn.Module):
    def __init__(self, c, tcr, shortcut=True) -> None:
        super().__init__()
        self.t_ch = int(c * tcr)
        self.c_ch = c - self.t_ch

        self.c_b = Bottleneck(self.c_ch, self.c_ch, shortcut=shortcut)
        self.t_b = MHSA_CGLU(self.t_ch)

        self.conv_fuse = Conv(c, c)

    def forward(self, x):
        cnn_branch, transformer_branch = x.split((self.c_ch, self.t_ch), 1)

        cnn_branch = self.c_b(cnn_branch)
        transformer_branch = self.t_b(transformer_branch)

        return self.conv_fuse(torch.cat([cnn_branch, transformer_branch], dim=1))


class CSP_PTB(nn.Module):
    """CSP-PTB(Partially Transformer Block)."""

    def __init__(self, c1, c2, n=1, tcr=0.25, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(PartiallyTransformerBlock(self.c, tcr, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


######################################## CSP-PTB(Partially Transformer Block) end ########################################

######################################## C2f_PPA Parallelized Patch-Aware Attention Module start ########################################
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = Conv(in_features, filters, act=False)
        self.c1 = Conv(filters, filters, 3)
        self.c2 = Conv(filters, filters, 3)
        self.c3 = Conv(filters, filters, 3)
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.silu = nn.SiLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x


class C3_PPA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PPA(c_, c_) for _ in range(n)))


class C2f_PPA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(PPA(self.c, self.c) for _ in range(n))


######################################## C2f_PPA Parallelized Patch-Aware Attention Module end ########################################

######################################## FeaturePyramidSharedConv Module start ########################################

class FeaturePyramidSharedConv(nn.Module):
    def __init__(self, c1, c2, dilations=[1, 3, 5]) -> None:
        super().__init__()

        c_ = c1 // 2  # hidden channels
        # 将输入通道数从 c1 减少到 c_
        self.cv1 = Conv(c1, c_, 1, 1)
        # 将多个特征图拼接后的通道数从 c_ * (1 + len(dilations)) 减少到 c2
        self.cv2 = Conv(c_ * (1 + len(dilations)), c2, 1, 1)
        # 共享的卷积层
        self.share_conv = nn.Conv2d(in_channels=c_, out_channels=c_, kernel_size=3, stride=1, padding=1, bias=False)
        self.dilations = dilations

    def forward(self, x):
        # 存储各个膨胀率下的特征图
        # 将输入 x 经过 cv1 卷积后的特征图添加到 y 中
        y = [self.cv1(x)]
        for dilation in self.dilations:
            # 对于每个膨胀率 dilation，使用 F.conv2d 函数对 y 中最后一个特征图进行卷积操作，并将其结果添加到 y 中
            y.append(F.conv2d(y[-1], weight=self.share_conv.weight, bias=None, dilation=dilation,
                              padding=(dilation * (3 - 1) + 1) // 2))
        #  将所有特征图在通道维度上拼接起来,将拼接后的特征图经过 cv2 卷积层，返回最终的输出.
        return self.cv2(torch.cat(y, 1))


######################################## FeaturePyramidSharedConv Module end ########################################

######################################## Edge information enhancement module start ########################################

class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :,
               0]


class EIEStem(nn.Module):
    def __init__(self, inc, hidc, ouc) -> None:
        super().__init__()

        self.conv1 = Conv(inc, hidc, 3, 2)
        self.sobel_branch = SobelConv(hidc)
        self.pool_branch = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
        )
        self.conv2 = Conv(hidc * 2, hidc, 3, 2)
        self.conv3 = Conv(hidc, ouc, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.sobel_branch(x), self.pool_branch(x)], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


# 用于边缘增强
class EIEM(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()

        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc * 2, inc, 1)
        self.conv2 = Conv(inc, ouc, 1)

    def forward(self, x):
        x_sobel = self.sobel_branch(x)
        x_conv = self.conv_branch(x)
        x_concat = torch.cat([x_sobel, x_conv], dim=1)
        x_feature = self.conv1(x_concat)
        x = self.conv2(x_feature + x)
        return x


class C3_EIEM(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(EIEM(c_, c_) for _ in range(n)))


class C2f_EIEM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EIEM(self.c, self.c) for _ in range(n))


######################################## Edge information enhancement module end ########################################

######################################## SPD-Conv start ########################################

# 与v5的Focus相同
class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2],
                       x[..., 1::2, ::2],
                       x[..., ::2, 1::2],
                       x[..., 1::2, 1::2]],
                      1)
        x = self.conv(x)
        return x


######################################## SPD-Conv end ########################################

######################################## Omni-Kernel Network for Image Restoration [AAAI-24] start ########################################

class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)

        ### fca ###
        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        ### sca ###
        x_sca = self.fgm(x_sca)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class CSPOmniKernel(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(x.size(1) * self.e), int(x.size(1) * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))


######################################## Omni-Kernel Network for Image Restoration [AAAI-24] end ########################################

######################################## Rectangular Self-Calibration Module [ECCV-24] start ########################################

class GetIndexOutput(nn.Module):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class PyramidPoolAgg_PCE(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RCA(nn.Module):
    def __init__(self, inp, kernel_size=1, ratio=2, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=3, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        # [N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w  # .repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather)  # [N, 1, C, 1]

        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc

        return out


class RCM(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=RCA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size,
                                       ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class multiRCM(nn.Module):
    def __init__(self, dim, n=3) -> None:
        super().__init__()
        self.mrcm = nn.Sequential(*[RCA(dim, 3, 2, square_kernel_size=1) for _ in range(n)])

    def forward(self, x):
        return self.mrcm(x)


class PyramidContextExtraction(nn.Module):
    def __init__(self, dim, n=3) -> None:
        super().__init__()

        self.dim = dim
        self.ppa = PyramidPoolAgg_PCE()
        self.rcm = nn.Sequential(*[RCA(sum(dim), 3, 2, square_kernel_size=1) for _ in range(n)])

    def forward(self, x):
        x = self.ppa(x)
        x = self.rcm(x)
        return torch.split(x, self.dim, dim=1)


class FuseBlockMulti(nn.Module):
    def __init__(
            self,
            inp: int,
    ) -> None:
        super(FuseBlockMulti, self).__init__()

        self.fuse1 = Conv(inp, inp, act=False)
        self.fuse2 = Conv(inp, inp, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        x_l, x_h = x
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out


class DynamicInterpolationFusion(nn.Module):
    def __init__(self, chn) -> None:
        super().__init__()
        self.conv = nn.Conv2d(chn[1], chn[0], kernel_size=1)

    def forward(self, x):
        return x[0] + self.conv(F.interpolate(x[1], size=x[0].size()[2:], mode='bilinear', align_corners=False))


######################################## Rectangular Self-Calibration Module [ECCV-24] end ########################################

######################################## RMT start ########################################

class C3_RetBlock(C3):
    def __init__(self, c1, c2, n=1, retention='chunk', num_heads=8, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.retention = retention
        self.Relpos = RelPos2d(c_, num_heads, 2, 4)
        self.m = nn.Sequential(*(RetBlock(retention, c_, num_heads, c_) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        b, c, h, w = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.retention == 'chunk')

        cv1 = self.cv1(x)
        for idx, layer in enumerate(self.m):
            if idx == 0:
                cv1 = layer(cv1.permute(0, 2, 3, 1), None, self.retention == 'chunk', rel_pos)
            else:
                cv1 = layer(cv1, None, self.retention == 'chunk', rel_pos)
        cv2 = self.cv2(x)
        return self.cv3(torch.cat((cv1.permute(0, 3, 1, 2), cv2), 1))


class C2f_RetBlock(C2f):
    def __init__(self, c1, c2, n=1, retention='chunk', num_heads=8, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.retention = retention
        self.Relpos = RelPos2d(self.c, num_heads, 2, 4)
        self.m = nn.ModuleList(RetBlock(retention, self.c, num_heads, self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        b, c, h, w = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.retention == 'chunk')

        y = list(self.cv1(x).chunk(2, 1))
        for layer in self.m:
            y.append(layer(y[-1].permute(0, 2, 3, 1), None, self.retention == 'chunk', rel_pos).permute(0, 3, 1, 2))
        return self.cv2(torch.cat(y, 1))


######################################## RMT end ########################################

######################################## Focus Diffusion Pyramid Network end ########################################
class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)

        self.dw_conv = nn.ModuleList(
            nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)

        x = x + feature
        return x


######################################## Focus Diffusion Pyramid Network end ########################################


class AxialFusionUnit(nn.Module):
    """ 整合空间注意力与特征融合的增强模块
    Args:
        dim (int): 输入通道数
        mlp_ratio (int): MLP扩展比例，默认4
        band_kernel (int): 条带卷积核大小，默认11
        square_kernel (int): 方形卷积核大小，默认3
        drop_path (float): DropPath概率，默认0
        layer_scale (float): 层缩放初始值，默认1e-6
        expansion_ratio (float): 通道压缩比例，默认0.5
    """

    def __init__(self, dim, mlp_ratio=4, band_kernel=11, square_kernel=3,
                 drop_path=0., layer_scale=1e-6, expansion_ratio=0.5):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        # ----------------- 空间注意力分支 -----------------
        # 局部特征提取
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, square_kernel, padding=square_kernel // 2, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # 动态通道增强 (新增1x1卷积)
        self.channel_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 16, dim, 1),
            nn.Sigmoid()
        )

        # 双轴向注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, (1, band_kernel),
                      padding=(0, band_kernel // 2), groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, dim, (band_kernel, 1),
                      padding=(band_kernel // 2, 0), groups=hidden_dim),
            nn.Sigmoid()
        )

        # ----------------- 特征融合分支 -----------------
        # 增强MLP (新增双1x1卷积)
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim * mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim * mlp_ratio, dim, 1)
        )

        # ----------------- 正则化组件 -----------------
        self.layer_scale = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.skip_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = self.skip_conv(x)

        # 空间注意力计算
        local_feat = self.local_conv(x)
        channel_attn = self.channel_proj(x)
        spatial_attn = self.spatial_gate(x)
        attn_feat = local_feat * channel_attn * spatial_attn

        # 特征融合
        fused_feat = self.mlp(attn_feat)

        # 残差连接
        if self.layer_scale is not None:
            fused_feat = fused_feat * self.layer_scale.view(1, -1, 1, 1)

        return shortcut + self.drop_path(fused_feat)


class DynamicSpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7], reduction=16):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)

        # 多分支卷积（不同kernel size）
        self.conv_branches = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])

        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, self.num_kernels, kernel_size=1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([max_pool, avg_pool], dim=1)

        # 生成各分支权重 (B×K×1×1)
        branch_weights = self.weight_generator(spatial_feat)

        # 多分支卷积 + 加权融合
        sa_maps = []
        for i, conv in enumerate(self.conv_branches):
            sa = conv(spatial_feat)  # 各分支输出 (B×1×H×W)
            weight = branch_weights[:, i:i + 1, :, :]  # 取对应分支权重
            sa_weighted = sa * weight.expand_as(sa)
            sa_maps.append(sa_weighted)

        # 求和并Sigmoid
        sa_fused = torch.sum(torch.stack(sa_maps, dim=0), dim=0)
        sa_weight = self.sigmoid(sa_fused)
        return sa_weight


class DCSAM(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_sizes=[3, 5, 7]):
        super().__init__()
        # 通道注意力（与原CSAM一致）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

        # 动态空间注意力
        self.spatial_attention = DynamicSpatialAttention(kernel_sizes, reduction)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        # 通道注意力
        ca_weight = self.channel_attention(x)
        x_ca = x * ca_weight

        # 动态空间注意力
        sa_weight = self.spatial_attention(x_ca)
        x_sa = x_ca * sa_weight

        return x_sa


class ContextGuideFusionModule_DCSAM(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.se = DCSAM(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


class ContextGuideFusionModule_noDCSAM(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        # self.se = DCSAM(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        # x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


class CGFM1(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        # self.se = DCSAM(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        # x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 * x1_weight, x1 * x0_weight], dim=1)


# class LA(nn.Module):
#     def __init__(self, dim=256, expansion=2, kernel_options=[1, 5, 7]):
#         super().__init__()
#         # 动态卷积选择器
#         self.dynamic_conv = nn.ModuleList([
#             nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim)
#             for k in kernel_options
#         ])
#         self.selector = nn.Linear(dim, len(kernel_options))
#
#         # 轻量通道注意力
#         self.channel_attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, max(4, dim // 32), 1, bias=False),  # 动态压缩
#             nn.GELU(),
#             nn.Conv2d(max(4, dim // 32), dim, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         # 轴向注意力优化
#         self.axial_attn = nn.Sequential(
#             nn.Conv2d(dim, dim, (1, kernel_options[0]), padding=(0, kernel_options[0] // 2)),
#             nn.Conv2d(dim, dim, (kernel_options[0], 1), padding=(kernel_options[0] // 2, 0)),
#             nn.Sigmoid()
#         )
#
#         # 高效特征融合
#         self.fusion = nn.Sequential(
#             nn.Conv2d(dim, dim * expansion, 1),
#             nn.GELU(),
#             nn.Conv2d(dim * expansion, dim, 1)
#         )
#
#     def forward(self, x):
#         # 动态卷积选择
#         gate = self.selector(x.mean([2, 3]))  # 全局平均池化后通过线性层
#         conv_idx = torch.argmax(gate, dim=1)  # 获取每个样本的最优卷积核索引
#         x_conv = sum(
#             self.dynamic_conv[i](x) * (conv_idx == i).float()[:, None, None, None]
#             for i in range(len(self.dynamic_conv))
#         )
#
#         # 复合注意力机制
#         channel_attn = self.channel_attn(x)  # 通道注意力
#         axial_attn = self.axial_attn(x_conv)  # 轴向注意力
#         attn = channel_attn * axial_attn  # 组合两种注意力
#
#         # 残差融合
#         return x + self.fusion(x * attn)


#
# class DAA(nn.Module):  # DynamicSparseAxialAttention
#     def __init__(self, dim=256, ratio=0.5, group=4):
#         super().__init__()
#         self.dim = dim
#         self.group = group
#         self.ratio = ratio
#
#         # 可学习稀疏控制器
#         self.sparse_controller = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, dim // 16, 1),
#             nn.ReLU(),
#             nn.Conv2d(dim // 16, 2, 1)
#         )
#
#         # 共享轴向卷积
#         self.axial_conv = nn.ModuleDict({
#             'h': nn.Conv2d(dim, dim, (1, 3), padding=(0, 1)),  # 水平方向卷积
#             'v': nn.Conv2d(dim, dim, (3, 1), padding=(1, 0))  # 垂直方向卷积
#         })
#
#         # 添加padding初始化
#         for conv in self.axial_conv.values():
#             conv.padding = (conv.kernel_size[0] // 2, conv.kernel_size[1] // 2)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         # 动态稀疏控制
#         control = self.sparse_controller(x)  # [B,2,1,1]
#         h_ratio = torch.sigmoid(control[:, 0]) * self.ratio
#         v_ratio = torch.sigmoid(control[:, 1]) * self.ratio
#
#         # 水平轴向处理
#         h_feat = self.axial_conv['h'](x)
#         h_mask = self._generate_mask(B, H, W, h_ratio, 'horizontal', x.device)
#         h_feat = h_feat * h_mask
#
#         # 垂直轴向处理
#         v_feat = self.axial_conv['v'](h_feat)
#         v_mask = self._generate_mask(B, H, W, v_ratio, 'vertical', x.device)
#         v_feat = v_feat * v_mask
#
#         return x + v_feat
#
#     def _generate_mask(self, B, H, W, ratio, direction, device):
#         """生成动态稀疏掩码
#         Args:
#             B: batch size
#             H: 特征图高度
#             W: 特征图宽度
#             ratio: 稀疏率
#             direction: 方向选择
#             device: 设备信息
#         """
#         k = int((H if direction == 'vertical' else W) * ratio.clamp(min=0.1, max=0.9))  # 限制稀疏率范围
#         indices = torch.randperm(H if direction == 'vertical' else W, device=device)[:k]
#         mask = torch.zeros(B, 1, H, W, device=device)
#         if direction == 'vertical':
#             mask[:, :, :, indices] = 1
#         else:
#             mask[:, :, indices, :] = 1
#         return mask


class DAA(nn.Module):
    def __init__(self, dim, ratio=0.5, group=4):
        super().__init__()
        self.dim = dim
        self.group = group
        self.ratio = ratio

        # 确保通道数能被分组数整除
        assert dim % group == 0, f"dim {dim} must be divisible by group {group}"

        # 可学习稀疏控制器
        self.sparse_controller = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 16, 2, 1)
        )

        # 共享轴向卷积（添加 padding）
        self.axial_conv = nn.ModuleDict({
            'h': nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=group),
            'v': nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=group)
        })

    def _generate_mask(self, B, H, W, ratio, direction):
        # 获取模型所在的设备
        device = next(self.parameters()).device

        k = int((H if direction == 'vertical' else W) * ratio)
        indices = torch.randperm(H if direction == 'vertical' else W)[:k]

        mask = torch.zeros(B, 1, H, W, device=device)
        if direction == 'horizontal':
            mask[:, :, :, indices] = 1  # 作用于宽度（第4维）
        else:
            mask[:, :, indices, :] = 1  # 作用于高度（第3维）
        return mask

    def forward(self, x):
        B, C, H, W = x.shape

        # 动态稀疏控制
        control = self.sparse_controller(x)  # [B,2,1,1]
        h_ratio = torch.sigmoid(control[:, 0]) * self.ratio
        v_ratio = torch.sigmoid(control[:, 1]) * self.ratio

        # 水平轴向处理
        h_feat = self.axial_conv['h'](x)
        h_mask = self._generate_mask(B, H, W, h_ratio, 'horizontal')
        h_feat = h_feat * h_mask

        # 垂直轴向处理
        v_feat = self.axial_conv['v'](h_feat)
        v_mask = self._generate_mask(B, H, W, v_ratio, 'vertical')
        v_feat = v_feat * v_mask

        return x + v_feat


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3,
                                   padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)


class LDCFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.channel_align = Conv(inc[0], inc[1], 1) if inc[0] != inc[1] else nn.Identity()

        # 动态门控生成器
        self.gate_gen = nn.Sequential(
            nn.Conv2d(inc[1] * 2, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x0, x1):
        x0 = self.channel_align(x0)
        gate = self.gate_gen(torch.cat([x0, x1], dim=1))
        return torch.cat([
            gate[:, 0:1] * x0 + (1 - gate[:, 0:1]) * x1,
            gate[:, 1:2] * x1 + (1 - gate[:, 1:2]) * x0
        ], dim=1)


class LDCFM_D(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.channel_align = Conv(inc[0], inc[1], 1) if inc[0] != inc[1] else nn.Identity()

        # 在L-DGF中替换标准卷积
        self.gate_gen = nn.Sequential(
            DepthwiseSeparableConv(inc[1] * 2, 8),
            nn.GELU(),
            DepthwiseSeparableConv(8, 2),
            nn.Sigmoid()
        )

    def forward(self, x0, x1):
        x0 = self.channel_align(x0)
        gate = self.gate_gen(torch.cat([x0, x1], dim=1))
        return torch.cat([
            gate[:, 0:1] * x0 + (1 - gate[:, 0:1]) * x1,
            gate[:, 1:2] * x1 + (1 - gate[:, 1:2]) * x0
        ], dim=1)


class GC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 几何特征提取
        self.geo_extract = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 3, padding=1),
            nn.GroupNorm(4, dim // 8)
        )

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(dim // 8, 4)

    def forward(self, x):
        B, C, H, W = x.shape
        x0 = x
        x1 = x
        g0 = self.geo_extract(x0).flatten(2)  # [B, C', N]
        g1 = self.geo_extract(x1).flatten(2)

        attn_out, _ = self.cross_attn(g0, g1, g1)
        attn_map = attn_out.view(B, -1, H, W)

        return x0 + x1 * attn_map


class ModalityInteractionChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # 通道统计量的融合
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels // 2, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, x1):
        # 输入：x0 (RGB), x1 (Depth)
        # 全局池化提取通道统计量
        max0 = F.max_pool2d(x0, kernel_size=x0.size()[2:])
        avg0 = F.avg_pool2d(x0, kernel_size=x0.size()[2:])
        max1 = F.max_pool2d(x1, kernel_size=x1.size()[2:])
        avg1 = F.avg_pool2d(x1, kernel_size=x1.size()[2:])

        # 融合两个模态的统计量
        fused_stat = torch.cat([max0 + avg0, max1 + avg1], dim=1)
        out = self.conv1(fused_stat)
        out = self.relu(out)
        out = self.conv2(out)
        return self.sigmoid(out)


class DynamicDualStreamSpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        # 多分支卷积（不同 kernel size）
        self.conv_branches = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        # 动态权重生成器（基于双流空间特征）
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, self.num_kernels, kernel_size=1),
            nn.Softmax(dim=1)  # 动态权重归一化
        )

    def forward(self, x0, x1):
        # 输入：x0 (RGB), x1 (Depth)
        # 提取双流空间特征（最大值和均值）
        max0 = F.max_pool2d(x0, kernel_size=x0.size()[2:])
        avg0 = F.avg_pool2d(x0, kernel_size=x0.size()[2:])
        max1 = F.max_pool2d(x1, kernel_size=x1.size()[2:])
        avg1 = F.avg_pool2d(x1, kernel_size=x1.size()[2:])
        spatial_feat = torch.cat([max0 + avg0, max1 + avg1], dim=1)

        # 生成各分支权重
        branch_weights = self.weight_generator(spatial_feat)

        # 多分支卷积 + 加权融合
        sa_maps = []
        for i, conv in enumerate(self.conv_branches):
            sa = conv(spatial_feat)
            weight = branch_weights[:, i:i + 1, :, :]
            sa_weighted = sa * weight.expand_as(sa)
            sa_maps.append(sa_weighted)
        sa_fused = sum(sa_maps)
        return sa_fused


class MGDAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.mica = ModalityInteractionChannelAttention(in_channels)
        self.ddsa = DynamicDualStreamSpatialAttention()

        # 特征解耦与重参数化
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x0, x1):
        # 输入：x0 (RGB), x1 (Depth)
        # 模态交互通道注意力
        ca_weight = self.mica(x0, x1)
        x_ca = x0 * ca_weight + x1 * (1 - ca_weight)  # 联合通道加权

        # 动态双流空间注意力
        sa_weight = self.ddsa(x0, x1)
        x_sa = x_ca * sa_weight

        # 特征解耦与重参数化
        x_out = x_sa * self.scale + self.shift
        return x_out + (x0 + x1)  # 残差连接增强梯度传递


class ContextGuideFusionModule_MGDAM(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.mghdam = MGDAM(inc[1])

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        # x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


class LA(nn.Module):
    def __init__(self, dim=512, mlp_ratio=2, band_kernel=7, expansion_ratio=0.25):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        # 轻量空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, (1, band_kernel),
                      padding=(0, band_kernel // 2), groups=hidden_dim),
            nn.Conv2d(hidden_dim, dim, (band_kernel, 1),
                      padding=(band_kernel // 2, 0), groups=hidden_dim),
            nn.Sigmoid()
        )

        # 高效通道增强
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 8, 16), 1),  # 动态压缩
            nn.GELU(),
            nn.Conv2d(max(dim // 8, 16), dim, 1),
            nn.Sigmoid()
        )

        # Bottleneck MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // mlp_ratio, dim, 1)
        )

    def forward(self, x):
        attn = self.channel_attn(x) * self.spatial_attn(x)
        return x + self.mlp(x * attn)


class CMGA(nn.Module):
    """基于跨模态引导的动态门控注意力"""

    def __init__(self, dim, reduction=8):
        super().__init__()
        # 如果 dim 是列表，取第一个元素
        if isinstance(dim, list):
            dim = dim[0]  # 假设 dim 列表的第一个元素是通道数
        else:
            dim = dim
        self.cross_guide = nn.Sequential(
            nn.Conv2d(2 * dim, dim // reduction, 1),
            nn.LayerNorm([dim // reduction, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(dim // reduction, 2, 1),
            nn.Softmax(dim=1)
        )

        self.channel_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )

    def forward(self, x):
        rgb_feat, depth_feat = x

        # 跨模态引导权重
        guide_weight = self.cross_guide(torch.cat([
            rgb_feat.mean(dim=1, keepdim=True),
            depth_feat.mean(dim=1, keepdim=True)
        ], dim=1))

        # 动态特征融合
        fused = guide_weight[:, 0:1] * self.channel_mixer(rgb_feat) + \
                guide_weight[:, 1:2] * depth_feat
        return fused


class n_DynamicSpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 5], reduction=8):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)

        self.conv_branches = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])

        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, self.num_kernels, 1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([max_pool, avg_pool], dim=1)

        branch_weights = self.weight_generator(spatial_feat)
        sa_maps = []
        for i, conv in enumerate(self.conv_branches):
            sa = conv(spatial_feat)
            weight = branch_weights[:, i:i + 1, :, :]
            sa_weighted = sa * weight.expand_as(sa)
            sa_maps.append(sa_weighted)

        sa_fused = torch.sum(torch.stack(sa_maps, dim=0), dim=0)
        sa_weight = self.sigmoid(sa_fused)
        return sa_weight


class n_DCSAM(nn.Module):
    def __init__(self, channel=512, reduction=8, kernel_sizes=[3, 5]):
        super().__init__()
        # 修正通道注意力结构
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = DynamicSpatialAttention(kernel_sizes, reduction)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        ca_weight = self.channel_attention(x)
        x_ca = x * ca_weight
        sa_weight = self.spatial_attention(x_ca)
        x_sa = x_ca * sa_weight
        return x_sa


class n_DCFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        # 调整压缩后的通道数为原输入的通道数之和
        self.compress_conv = nn.Conv2d(inc[1] * 2, inc[1] * 2, kernel_size=1, bias=False)  # 输出通道数为 inc[1] * 2

        self.se = DCSAM(inc[1] * 2, reduction=8, kernel_sizes=[3, 5])  # 注意：DCSAM的输入通道数也要调整

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        x_concat = torch.cat([x0, x1], dim=1)
        x_concat = self.compress_conv(x_concat)  # 通道数为 inc[1] * 2（例如 192*2=384）

        x_concat = self.se(x_concat)  # 确保 DCSAM 的输入通道数与 compress_conv 的输出一致

        # 分割后的通道数总和必须等于 x_concat.size(1)
        split_sizes = [x0.size(1), x1.size(1)]  # 总和为 192+192=384
        x0_weight, x1_weight = torch.split(x_concat, split_sizes, dim=1)

        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


# ++++++++++++++++++++++++++++以下为消融实验部分+++++++++++++++++++++++++++++++++++ #

# ###############融合模块消融################ #


# 1. 只采用融合框架，不使用注意力 (CGFM_1)
class CGFM_1(nn.Module):  # ContextGuideFusionModule_1
    def __init__(self, inc) -> None:
        super().__init__()
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        return torch.cat([x0 + x1, x1 + x0], dim=1)  # 原始融合框架


# 2. 融合框架+通道注意力 (CGFM_12)
class DCSAM_ChannelOnly(nn.Module):  # 仅保留通道注意力的变体
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_weight = self.channel_attention(x)
        return x * ca_weight  # 仅应用通道注意力


class CGFM_12(nn.Module):  # ContextGuideFusionModule_12
    def __init__(self, inc) -> None:
        super().__init__()
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)
        self.se = DCSAM_ChannelOnly(inc[1] * 2)  # 仅使用通道注意力

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        x_concat = torch.cat([x0, x1], dim=1)
        x_concat = self.se(x_concat)  # 仅应用通道注意力
        x0_part, x1_part = torch.split(x_concat, [x0.size(1), x1.size(1)], dim=1)
        return torch.cat([x0 + x1_part, x1 + x0_part], dim=1)


# 3. 融合框架+空间注意力 (CGFM_13)
class DCSAM_SpatialOnly(DynamicSpatialAttention):  # 仅保留空间注意力的变体
    def __init__(self, kernel_sizes=[3, 5, 7], reduction=16):
        super().__init__(kernel_sizes, reduction)

    def forward(self, x):
        sa_weight = super().forward(x)
        return x * sa_weight  # 直接返回空间注意力结果


class CGFM_13(nn.Module):  # ContextGuideFusionModule_13
    def __init__(self, inc) -> None:
        super().__init__()
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)
        self.se = DCSAM_SpatialOnly()  # 仅使用空间注意力

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        x_concat = torch.cat([x0, x1], dim=1)
        x_concat = self.se(x_concat)  # 仅应用空间注意力
        x0_part, x1_part = torch.split(x_concat, [x0.size(1), x1.size(1)], dim=1)
        return torch.cat([x0 + x1_part, x1 + x0_part], dim=1)


# 4. 调整顺序的完整DCSAM (CGFM_132)
class DCSAM_ReverseOrder(DCSAM):
    def forward(self, x):
        # 先空间注意力
        sa_weight = self.spatial_attention(x)
        x_sa = x * sa_weight

        # 再通道注意力
        ca_weight = self.channel_attention(x_sa)
        x_ca = x_sa * ca_weight

        return x_ca


class CGFM_132(ContextGuideFusionModule_DCSAM):  # ContextGuideFusionModule_132
    def __init__(self, inc) -> None:
        super().__init__(inc)
        self.se = DCSAM_ReverseOrder(inc[1] * 2)  # 使用调整顺序的DCSAM


# 5. Add+通道注意力 (Add_2)
class Add_2(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.channel_attention = DCSAM_ChannelOnly(inc[1])

    def forward(self, x):
        x0, x1 = x
        x_add = x0 + x1
        x_att = self.channel_attention(x_add)
        return x_att


# 6. Add+空间注意力 (Add_3)
class Add_3(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.spatial_attention = DynamicSpatialAttention()

    def forward(self, x):
        x0, x1 = x
        x_add = x0 + x1
        sa_weight = self.spatial_attention(x_add)
        return x_add * sa_weight


# 7. Add+完整注意力 (Add_23)
class Add_23(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.attention = DCSAM(inc[1])

    def forward(self, x):
        x0, x1 = x
        x_add = x0 + x1
        return self.attention(x_add)


# 8. Add+逆序注意力 (Add_32)
class Add_32(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.attention = DCSAM_ReverseOrder(inc[1])

    def forward(self, x):
        x0, x1 = x
        x_add = x0 + x1
        return self.attention(x_add)


# 9. 融合框架+通道注意力和空间注意力并行 (CGFM_1_2_3)
class DCSAM_Parallel(nn.Module):
    """通道和空间注意力并行计算版本"""

    def __init__(self, channel=512, reduction=16, kernel_sizes=[3, 5, 7]):
        super().__init__()
        # 通道注意力（保持原结构）
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力（保持原DynamicSpatialAttention结构）
        self.sa = DynamicSpatialAttention(kernel_sizes, reduction)

    def forward(self, x):
        # 并行计算
        ca_weight = self.ca(x)
        sa_weight = self.sa(x)
        # 合并方式：先通道加权，再空间加权（与相加/相乘等效）
        return x * ca_weight * sa_weight


class CGFM_1_2_3(ContextGuideFusionModule_DCSAM):
    def __init__(self, inc):
        super().__init__(inc)
        self.se = DCSAM_Parallel(inc[1] * 2)  # 使用并行版本


# 10.Add+空间、通道并行（Add_1_2_3）
class Add_1_2_3(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)
        self.attention = DCSAM_Parallel(inc[1])  # 使用并行版本

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        return self.attention(x0 + x1)


# #
# # class DynamicSpatialAttention(nn.Module):
# #     def __init__(self, kernel_sizes=[3, 5, 7], reduction=16):
# #         super().__init__()
# #         self.kernel_sizes = kernel_sizes
# #         self.num_kernels = len(kernel_sizes)
# #
# #         # 多分支卷积（不同kernel size）
# #         self.conv_branches = nn.ModuleList([
# #             nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
# #             for k in kernel_sizes
# #         ])
# #
# #         # 动态权重生成器
# #         self.weight_generator = nn.Sequential(
# #             nn.AdaptiveAvgPool2d(1),
# #             nn.Conv2d(2, self.num_kernels, kernel_size=1),
# #             nn.Sigmoid()
# #         )
# #
# #         self.sigmoid = nn.Sigmoid()
# #
# #     def forward(self, x):
# #         max_pool = torch.max(x, dim=1, keepdim=True)[0]
# #         avg_pool = torch.mean(x, dim=1, keepdim=True)
# #         spatial_feat = torch.cat([max_pool, avg_pool], dim=1)
# #
# #         # 生成各分支权重 (B×K×1×1)
# #         branch_weights = self.weight_generator(spatial_feat)
# #
# #         # 多分支卷积 + 加权融合
# #         sa_maps = []
# #         for i, conv in enumerate(self.conv_branches):
# #             sa = conv(spatial_feat)  # 各分支输出 (B×1×H×W)
# #             weight = branch_weights[:, i:i + 1, :, :]  # 取对应分支权重
# #             sa_weighted = sa * weight.expand_as(sa)
# #             sa_maps.append(sa_weighted)
# #
# #         # 求和并Sigmoid
# #         sa_fused = torch.sum(torch.stack(sa_maps, dim=0), dim=0)
# #         sa_weight = self.sigmoid(sa_fused)
# #         return sa_weight
#
#
# class DCA(nn.Module):  # DCSAM只有通道
#     def __init__(self, channel, reduction=16, kernel_sizes=[3, 5, 7]):
#         super().__init__()
#         # 通道注意力（与原CSAM一致）
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         # 动态空间注意力
#         # self.spatial_attention = DynamicSpatialAttention(kernel_sizes, reduction)
#
#     #     self.init_weights()
#     #
#     # def init_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv2d):
#     #             init.kaiming_normal_(m.weight, mode='fan_out')
#     #             if m.bias is not None:
#     #                 init.constant_(m.bias, 0)
#     #         elif isinstance(m, nn.BatchNorm2d):
#     #             init.constant_(m.weight, 1)
#     #             init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # 通道注意力
#         ca_weight = self.channel_attention(x)
#         x_ca = x * ca_weight
#
#         # # 动态空间注意力
#         # sa_weight = self.spatial_attention(x_ca)
#         # x_sa = x_ca * sa_weight
#
#         return x_ca
#
#
# class DSA(nn.Module):  # DCSAM只有空间
#     def __init__(self, channel=512, reduction=16, kernel_sizes=[3, 5, 7]):
#         super().__init__()
#         # 通道注意力（与原CSAM一致）
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         # 动态空间注意力
#         self.spatial_attention = DynamicSpatialAttention(kernel_sizes, reduction)
#
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # # 通道注意力
#         # ca_weight = self.channel_attention(x)
#         # x_ca = x * ca_weight
#
#         # 动态空间注意力
#         sa_weight = self.spatial_attention(x)
#         x_sa = x * sa_weight
#
#         return x_sa
#
#
# class DCSAM_32(nn.Module):  # DCSAM先空间后通道
#     def __init__(self, channel=512, reduction=16, kernel_sizes=[3, 5, 7]):
#         super().__init__()
#         # 通道注意力（与原CSAM一致）
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         # 动态空间注意力
#         self.spatial_attention = DynamicSpatialAttention(kernel_sizes, reduction)
#
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # # 通道注意力
#         # ca_weight = self.channel_attention(x)
#         # x_ca = x * ca_weight
#
#         # 动态空间注意力
#         sa_weight = self.spatial_attention(x)
#         x_sa = x * sa_weight
#
#         ca_weight = self.channel_attention(x_sa)
#         x_ca = x_sa * ca_weight
#
#         return x_ca
#
#
# class CGFM_1(nn.Module):  # 无DCSAM模块，只有融合架构
#     def __init__(self, inc) -> None:
#         super().__init__()
#
#         self.adjust_conv = nn.Identity()
#         if inc[0] != inc[1]:
#             self.adjust_conv = Conv(inc[0], inc[1], k=1)
#
#         # self.attn = DCSAM(inc[1] * 2)
#
#     def forward(self, x):
#         x0, x1 = x
#         x0 = self.adjust_conv(x0)
#
#         x_concat = torch.cat([x0, x1], dim=1)  # n c h w
#         # x_concat = self.attn(x_concat)
#         x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
#         x0_weight = x0 * x0_weight
#         x1_weight = x1 * x1_weight
#         return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)
#
#
# class CGFM_12(nn.Module):  # 融合架构+DCA
#     def __init__(self, inc) -> None:
#         super().__init__()
#
#         self.adjust_conv = nn.Identity()
#         if inc[0] != inc[1]:
#             self.adjust_conv = Conv(inc[0], inc[1], k=1)
#
#         self.attn = DCA(inc[1] * 2)
#
#     def forward(self, x):
#         x0, x1 = x
#         x0 = self.adjust_conv(x0)
#
#         x_concat = torch.cat([x0, x1], dim=1)  # n c h w
#         x_concat = self.attn(x_concat)
#         x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
#         x0_weight = x0 * x0_weight
#         x1_weight = x1 * x1_weight
#         return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)
#
#
# class CGFM_13(nn.Module):  # 融合架构+DSA
#     def __init__(self, inc) -> None:
#         super().__init__()
#
#         self.adjust_conv = nn.Identity()
#         if inc[0] != inc[1]:
#             self.adjust_conv = Conv(inc[0], inc[1], k=1)
#
#         self.attn = DSA(inc[1] * 2)
#
#     def forward(self, x):
#         x0, x1 = x
#         x0 = self.adjust_conv(x0)
#
#         x_concat = torch.cat([x0, x1], dim=1)  # n c h w
#         x_concat = self.attn(x_concat)
#         x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
#         x0_weight = x0 * x0_weight
#         x1_weight = x1 * x1_weight
#         return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)
#
#
# class Add_2(nn.Module):  # Add+DCA
#     def __init__(self, inc) -> None:
#         super().__init__()
#
#         # self.adjust_conv = nn.Identity()
#         # if inc[0] != inc[1]:
#         #     self.adjust_conv = Conv(inc[0], inc[1], k=1)
#
#         self.attn = DCA(inc)
#
#     def forward(self, x):
#         x0, x1 = x
#
#         return self.attn(torch.add(x0, x1))
#
#
# class Add_3(nn.Module):  # Add+DSA
#     def __init__(self, inc) -> None:
#         super().__init__()
#
#         # self.adjust_conv = nn.Identity()
#         # if inc[0] != inc[1]:
#         #     self.adjust_conv = Conv(inc[0], inc[1], k=1)
#
#         self.attn = DSA(inc[1] * 2)
#
#     def forward(self, x):
#         x0, x1 = x
#
#         return self.attn(torch.add(x0, x1))
#
#
# class Add_23(nn.Module):  # Add+DCA+DSA
#     def __init__(self, inc) -> None:
#         super().__init__()
#
#         # self.adjust_conv = nn.Identity()
#         # if inc[0] != inc[1]:
#         #     self.adjust_conv = Conv(inc[0], inc[1], k=1)
#
#         self.attn = DCSAM(inc[1] * 2)
#
#     def forward(self, x):
#         x0, x1 = x
#
#         return self.attn(torch.add(x0, x1))
#
#
# class Add_32(nn.Module):  # Add+DSA+DCA
#     def __init__(self, inc) -> None:
#         super().__init__()
#
#         # self.adjust_conv = nn.Identity()
#         # if inc[0] != inc[1]:
#         #     self.adjust_conv = Conv(inc[0], inc[1], k=1)
#
#         self.attn = DCSAM_32(inc[1] * 2)
#
#     def forward(self, x):
#         x0, x1 = x
#
#         return self.attn(torch.add(x0, x1))
#
#
# class CGFM_132(nn.Module):  # 融合架构+DSA+DCA
#     def __init__(self, inc) -> None:
#         super().__init__()
#
#         self.adjust_conv = nn.Identity()
#         if inc[0] != inc[1]:
#             self.adjust_conv = Conv(inc[0], inc[1], k=1)
#
#         self.attn = DCSAM_32(inc[1] * 2)
#
#     def forward(self, x):
#         x0, x1 = x
#         x0 = self.adjust_conv(x0)
#
#         x_concat = torch.cat([x0, x1], dim=1)  # n c h w
#         x_concat = self.attn(x_concat)
#         x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
#         x0_weight = x0 * x0_weight
#         x1_weight = x1 * x1_weight
#         return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


# ##########AAF消融############# #

# 1.移除空间注意力（仅保留通道注意力和 MLP）
class LA_1M(nn.Module):  # LA_NoSpatialAttn
    def __init__(self, dim=512, mlp_ratio=2, band_kernel=7, expansion_ratio=0.25):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        # 移除空间注意力
        self.spatial_attn = None  # 或直接删除

        # 保留通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 8, 16), 1),
            nn.GELU(),
            nn.Conv2d(max(dim // 8, 16), dim, 1),
            nn.Sigmoid()
        )

        # 保留 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // mlp_ratio, dim, 1)
        )

    def forward(self, x):
        channel_attn = self.channel_attn(x)
        # 因为空间注意力被移除，attn 只保留通道注意力
        return x + self.mlp(x * channel_attn)


# 2.移除通道注意力（仅保留空间注意力和 MLP）
class LA_2M(nn.Module):  # LA_NoChannelAttn
    def __init__(self, dim=512, mlp_ratio=2, band_kernel=7, expansion_ratio=0.25):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        # 保留空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, (1, band_kernel),
                      padding=(0, band_kernel // 2), groups=hidden_dim),
            nn.Conv2d(hidden_dim, dim, (band_kernel, 1),
                      padding=(band_kernel // 2, 0), groups=hidden_dim),
            nn.Sigmoid()
        )

        # 移除通道注意力
        self.channel_attn = None

        # 保留 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // mlp_ratio, dim, 1)
        )

    def forward(self, x):
        spatial_attn = self.spatial_attn(x)
        # 因为通道注意力被移除，attn 只保留空间注意力
        return x + self.mlp(x * spatial_attn)


# 3.移除 MLP（仅保留注意力模块）
class LA_12(nn.Module):  # LA_NoMLP
    def __init__(self, dim=512, mlp_ratio=2, band_kernel=7, expansion_ratio=0.25):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        # 保留空间和通道注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, (1, band_kernel),
                      padding=(0, band_kernel // 2), groups=hidden_dim),
            nn.Conv2d(hidden_dim, dim, (band_kernel, 1),
                      padding=(band_kernel // 2, 0), groups=hidden_dim),
            nn.Sigmoid()
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 8, 16), 1),
            nn.GELU(),
            nn.Conv2d(max(dim // 8, 16), dim, 1),
            nn.Sigmoid()
        )

        # 移除 MLP
        self.mlp = None

    def forward(self, x):
        spatial_attn = self.spatial_attn(x)
        channel_attn = self.channel_attn(x)
        attn = spatial_attn * channel_attn
        # 因为 MLP 被移除，直接返回原始输入乘以注意力
        return x * attn  # 无残差连接（需根据原始模型调整）


# 4.同时移除空间和通道注意力（仅保留 MLP）
class LA_M(nn.Module):
    def __init__(self, dim=512, mlp_ratio=2, band_kernel=7, expansion_ratio=0.25):
        super().__init__()
        # 移除所有注意力模块
        self.spatial_attn = None
        self.channel_attn = None

        # 保留 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // mlp_ratio, dim, 1)
        )

    def forward(self, x):
        # 直接使用 MLP，无注意力机制
        return x + self.mlp(x)  # 保留残差连接


# 6.空间注意力先 → 通道注意力后
class LA_2_1_M(nn.Module):
    def __init__(self, dim=512, mlp_ratio=2, band_kernel=7, expansion_ratio=0.25):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        # 轻量空间注意力（保持不变）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, (1, band_kernel),
                      padding=(0, band_kernel // 2), groups=hidden_dim),
            nn.Conv2d(hidden_dim, dim, (band_kernel, 1),
                      padding=(band_kernel // 2, 0), groups=hidden_dim),
            nn.Sigmoid()
        )

        # 高效通道增强（保持不变）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 8, 16), 1),
            nn.GELU(),
            nn.Conv2d(max(dim // 8, 16), dim, 1),
            nn.Sigmoid()
        )

        # Bottleneck MLP（保持不变）
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // mlp_ratio, dim, 1)
        )

    def forward(self, x):
        # 空间注意力先
        spatial_attn = self.spatial_attn(x)
        x_spatial = x * spatial_attn  # 应用空间注意力

        # 通道注意力的输入是空间处理后的特征（x_spatial）
        channel_attn = self.channel_attn(x_spatial)

        # 总注意力是两者的乘积
        total_attn = spatial_attn * channel_attn

        return x + self.mlp(x * total_attn)


# 7. 通道注意力先 → 空间注意力后
class LA_1_2_M(nn.Module):
    def __init__(self, dim=512, mlp_ratio=2, band_kernel=7, expansion_ratio=0.25):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        # 轻量空间注意力（保持不变）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, (1, band_kernel),
                      padding=(0, band_kernel // 2), groups=hidden_dim),
            nn.Conv2d(hidden_dim, dim, (band_kernel, 1),
                      padding=(band_kernel // 2, 0), groups=hidden_dim),
            nn.Sigmoid()
        )

        # 高效通道增强（保持不变）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 8, 16), 1),
            nn.GELU(),
            nn.Conv2d(max(dim // 8, 16), dim, 1),
            nn.Sigmoid()
        )

        # Bottleneck MLP（保持不变）
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // mlp_ratio, dim, 1)
        )

    def forward(self, x):
        # 通道注意力先
        channel_attn = self.channel_attn(x)
        x_channel = x * channel_attn  # 应用通道注意力

        # 空间注意力的输入是通道处理后的特征（x_channel）
        spatial_attn = self.spatial_attn(x_channel)

        # 总注意力是两者的乘积
        total_attn = channel_attn * spatial_attn

        return x + self.mlp(x * total_attn)


# IR

######################################## EfficientNet-MBConv start ########################################
class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


class MBConv(nn.Module):
    def __init__(self, inc, ouc, shortcut=True, e=4, dropout=0.1) -> None:
        super().__init__()
        midc = inc * e
        self.conv_pw_1 = Conv(inc, midc, 1)
        self.conv_dw_1 = Conv(midc, midc, 3, g=midc)
        self.effective_se = EffectiveSEModule(midc)
        self.conv1 = Conv(midc, ouc, 1, act=False)
        self.dropout = nn.Dropout2d(p=dropout)
        self.add = shortcut and inc == ouc

    def forward(self, x):
        return x + self.dropout(
            self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x))))) if self.add else self.dropout(
            self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))


class C3_EMBC(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MBConv(c_, c_, shortcut) for _ in range(n)))


class C2f_EMBC(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MBConv(self.c, self.c, shortcut) for _ in range(n))


######################################## EfficientNet-MBConv end ########################################


######################################## BIFPN begin ########################################

class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion
        self.inc_list = inc_list  # 保存输入通道列表

        if self.fusion == 'bifpn':
            # 为每个输入通道创建独立的1x1卷积用于通道对齐
            self.align_convs = nn.ModuleList([
                Conv(inc, max(inc_list), 1) if inc != max(inc_list) else nn.Identity()
                for inc in inc_list
            ])
            # 后续正常
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)

    def forward(self, x):
        # 首先对齐所有输入的通道数（如果是bifpn模式）
        if self.fusion == 'bifpn':
            max_channels = max(self.inc_list)
            aligned_x = []
            for i, xi in enumerate(x):
                if xi.size(1) != max_channels:
                    xi = self.align_convs[i](xi)
                aligned_x.append(xi)
            x = aligned_x

        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)


######################################## BIFPN end ########################################


######################################## Efficient Multi-Branch&Scale FPN start ########################################

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                Conv(self.in_channels, self.in_channels, kernel_size, s=stride, g=self.in_channels)
            )
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        return outputs


class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], stride=1, expansion_factor=2,
                 dw_parallel=True, add=True):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            Conv(self.in_channels, self.ex_channels, 1)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            Conv(self.combined_channels, self.out_channels, 1, act=False)
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class CSP_MSCB(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 5], shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(MSCB(self.c, self.c, kernel_sizes=kernel_sizes) for _ in range(n))


######################################## Multi-Branch&Scale-FPN end ########################################

######################################## SPPF with LSKA start ########################################
class LSKA(nn.Module):
    # Large-Separable-Kernel-Attention
    # https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/tree/main
    def __init__(self, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class SPPF_LSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))


######################################## SPPF with LSKA end ########################################

######################################## Cross-Scale Mutil-Head Self-Attention start ########################################

class CSMHSA(nn.Module):
    def __init__(self, n_dims, heads=8):
        super(CSMHSA, self).__init__()

        self.heads = heads
        self.query = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n_dims[0], n_dims[1], kernel_size=1)
        )
        self.key = nn.Conv2d(n_dims[1], n_dims[1], kernel_size=1)
        self.value = nn.Conv2d(n_dims[1], n_dims[1], kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_high, x_low = x
        n_batch, C, width, height = x_low.size()
        q = self.query(x_high).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x_low).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x_low).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        attention = self.softmax(content_content)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out


######################################## Cross-Scale Mutil-Head Self-Attention end ########################################

######################################## StartNet end ########################################
class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x


class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class Star_Block_CAA(Star_Block):
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)

        self.attention = CAA(mlp_ratio * dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x


class C3_Star(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block(c_) for _ in range(n)))


class C2f_Star(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block(self.c) for _ in range(n))


class C3_Star_CAA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block_CAA(c_) for _ in range(n)))


class C2f_Star_CAA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_CAA(self.c) for _ in range(n))


######################################## StartNet end ########################################


class IFM(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        # 通道对齐
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        # 特征交互模块
        self.interaction = nn.Sequential(
            Conv(inc[1] * 2, inc[1] // 2, k=1),  # 通道压缩
            Conv(inc[1] // 2, inc[1] * 2, k=3, p=1),  # 空间交互
            nn.Sigmoid()  # 权重归一化
        )

        # 残差连接参数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        # 生成交互权重
        x_concat = torch.cat([x0, x1], dim=1)
        interaction_weights = self.interaction(x_concat)

        # 拆分权重
        w0, w1 = torch.split(interaction_weights, [x0.size(1), x1.size(1)], dim=1)

        # 双路残差融合
        fused_x0 = x0 + self.alpha * (w1 * x1)
        fused_x1 = x1 + self.beta * (w0 * x0)

        # 特征整合输出
        return fused_x0 + fused_x1


# class PCE4(nn.Module): # PyramidContextExtraction4
#     def __init__(self, dim, n=3) -> None:
#         super().__init__()
#         self.dim = dim  # dim should be a list of 4 integers
#         self.ppa = PyramidPoolAgg_PCE()
#         self.rcm = nn.Sequential(*[RCA(sum(dim), 3, 2, square_kernel_size=1) for _ in range(n)])
#
#     def forward(self, x):
#         x = self.ppa(x)
#         x = self.rcm(x)
#         return torch.split(x, self.dim, dim=1)  # split into 4 outputs

class PCE4(nn.Module):  # PyramidContextExtraction4
    def __init__(self, dim, n=4) -> None:  # 修改1: 默认n=4支持4层
        super().__init__()
        self.dim = dim
        self.ppa = PyramidPoolAgg_PCE4()
        # 修改2: 使用n而不是固定值3
        self.rcm = nn.Sequential(*[RCA4(sum(dim), 3, 2, square_kernel_size=1) for _ in range(n)])

    def forward(self, x):
        x = self.ppa(x)
        x = self.rcm(x)
        # 修改3: 使用self.dim而不是固定值
        return torch.split(x, self.dim, dim=1)


class PyramidPoolAgg_PCE4(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        # 修改4: 支持任意数量的输入层
        if not inputs:
            return inputs

        # 使用最后一个输入确定尺寸
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        # 修改5: 处理所有输入层
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class RCA4(nn.Module):
    def __init__(self, inp, kernel_size=1, ratio=2, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=3, relu=True):
        super(RCA4, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w
        ge = self.excite(x_gather)
        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc
        return out


class CGFM(nn.Module):  # ContextGuideFusionModule_noDCSAM
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        # self.se = DCSAM(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        # x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


class DCGFM(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        # 第一层融合模块
        self.fusion1 = CGFM(inc)
        # 第二层融合模块（输入通道为[inc[1], inc[1]]）
        self.fusion2 = CGFM([inc[1], inc[1]])

    def forward(self, x):
        x0, x1 = x

        # 第一层融合
        fused = self.fusion1((x0, x1))  # 输出通道数: 2*inc[1]

        # 将融合结果拆分为两部分
        split_size = fused.size()[1] // 2
        part1, part2 = torch.split(fused, split_size, dim=1)

        # 第二层融合
        final_fused = self.fusion2((part1, part2))

        return final_fused


class C2f_Star_RGB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_RGB(self.c) for _ in range(n))


class C2f_Star_Depth(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_Depth(self.c) for _ in range(n))


# RGB流：增强纹理特征提取
class Star_Block_RGB(Star_Block):
    def __init__(self, dim, mlp_ratio=3):
        super().__init__(dim, mlp_ratio)
        # 增加纹理增强模块
        self.texture_enhance = nn.Sequential(
            Conv(dim, dim // 4, 3),  # 降维
            nn.AvgPool2d(3, stride=1, padding=1),  # 纹理提取
            Conv(dim // 4, dim, 1)  # 恢复维度
        )

    def forward(self, x):
        x = super().forward(x)
        return x + self.texture_enhance(x)  # 残差纹理增强


class Star_Block_Depth(Star_Block):
    def __init__(self, dim, mlp_ratio=2):  # 降低扩展率
        super().__init__(dim, mlp_ratio)
        # 边缘检测分支
        self.edge_detect = nn.Sequential(
            Conv(dim, dim, 3, act=False),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # 深度卷积
        )

    def forward(self, x):
        edge = self.edge_detect(x)
        return super().forward(x) + edge  # 增强边缘特征


class Enhanced_Star_Block(Star_Block):
    def __init__(self, dim, mlp_ratio=3):
        super().__init__(dim, mlp_ratio)
        # 多尺度特征融合（移除了 dilation 参数）
        self.multiscale = nn.Sequential(
            Conv(dim, dim // 4, 1),  # 1x1卷积降维
            Conv(dim // 4, dim // 4, 3),  # 普通3x3卷积（移除了 dilation）
            Conv(dim // 4, dim, 1)  # 1x1卷积恢复维度
        )
        # 轻量级注意力
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            Conv(dim, dim // 4, 1, act=nn.Hardswish()),  # 1x1卷积降维 + Hardswish激活
            Conv(dim // 4, dim, 1, act=nn.Sigmoid())  # 1x1卷积恢复维度 + Sigmoid激活
        )

    def forward(self, x):
        # 原始Star_Block处理
        res = super().forward(x)

        # 多尺度特征提取
        multiscale_feat = self.multiscale(x)

        # 注意力权重计算
        attn_weights = self.attn(res + multiscale_feat)

        # 特征融合
        return res * attn_weights + multiscale_feat


class Star_Block_RGB2(Star_Block):
    """频率域增强的Star Block (RGB专用)"""

    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__(dim, mlp_ratio, drop_path)
        # 频率增强模块
        self.freq_enhance = FreqEnhance(dim)

    def forward(self, x):
        inputw = x
        x = self.dwconv(x)

        # 频率增强分支
        freq_feat = self.freq_enhance(inputw)

        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return inputw + self.drop_path(x) + freq_feat


class FreqEnhance(nn.Module):
    """轻量化频率域增强模块"""

    def __init__(self, dim):
        super().__init__()
        self.low_freq_enhance = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            Conv(dim, dim // 4, 1),
            nn.ReLU6(),
            Conv(dim // 4, dim, 1))

        self.high_freq_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            Conv(dim, dim, 1))

    def forward(self, x):
        # 低频增强
        low_freq = self.low_freq_enhance(x)

        # 高频增强 (通过残差获取高频)
        high_freq = x - F.avg_pool2d(x, 3, stride=1, padding=1)
        high_freq = self.high_freq_enhance(high_freq)

        return low_freq + high_freq


class Star_Block_Depth2(Star_Block):
    """几何特征增强的Star Block (Depth专用)"""

    def __init__(self, dim, mlp_ratio=2, drop_path=0.):  # 降低mlp_ratio保持轻量
        super().__init__(dim, mlp_ratio, drop_path)

        # 几何特征分支
        self.geo_branch = GeoPerception(dim)

        # 替换原始dwconv2为分组几何卷积
        del self.dwconv2
        self.dwconv2 = GeoConv(dim, dim, groups=dim)

    def forward(self, x):
        inputw = x
        x = self.dwconv(x)

        # 几何特征提取
        geo_feat = self.geo_branch(inputw)

        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return inputw + self.drop_path(x) + geo_feat


class GeoPerception(nn.Module):
    """轻量化几何特征提取模块"""

    def __init__(self, dim):
        super().__init__()
        self.convs = nn.ModuleList([
            # 方向卷积组
            nn.Conv2d(dim // 4, dim // 4, (1, 5), padding=(0, 2), groups=dim // 4),
            nn.Conv2d(dim // 4, dim // 4, (5, 1), padding=(2, 0), groups=dim // 4),
            # 局部和全局特征
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, dilation=1, groups=dim // 4),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=3, dilation=3, groups=dim // 4)
        ])
        self.fuse = Conv(dim, dim, 1, act=False)

    def forward(self, x):
        # 分组处理特征
        chunks = torch.chunk(x, 4, dim=1)
        geo_feats = [conv(chunk) for conv, chunk in zip(self.convs, chunks)]
        return self.fuse(torch.cat(geo_feats, dim=1))


class GeoConv(nn.Module):
    """几何感知卷积"""

    def __init__(self, in_ch, out_ch, groups=1):
        super().__init__()
        self.conv_h = nn.Conv2d(in_ch // 2, out_ch // 2, (5, 1), padding=(2, 0), groups=groups // 2)
        self.conv_v = nn.Conv2d(in_ch // 2, out_ch // 2, (1, 5), padding=(0, 2), groups=groups // 2)

    def forward(self, x):
        h, v = torch.chunk(x, 2, dim=1)
        return torch.cat([self.conv_h(h), self.conv_v(v)], dim=1)


class Star_Block_Shuffle(Star_Block):
    """特征重组增强的Star Block"""

    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__(dim, mlp_ratio, drop_path)
        # 设置分组数
        self.groups = 4

        # 修正空间注意力模块
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),  # 降维减少计算量
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4),  # 保持分组数等于输入通道数
            Conv(dim // 4, 1, 1),  # 最后压缩到单通道
            nn.Sigmoid()
        )

    def manual_channel_shuffle(self, x):
        """手动实现的通道重排操作，兼容CUDA"""
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # 重塑张量以进行分组
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # 转置维度以进行重排
        x = x.transpose(1, 2).contiguous()

        # 重塑回原始形状
        return x.view(batch_size, -1, height, width)

    def forward(self, x):
        # 保存原始输入
        x_orig = x

        # 主路径处理
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))

        # 残差连接 + DropPath
        x_res = x_orig + self.drop_path(x)

        # 特征重组 - 使用手动实现的通道重排
        x_res = self.manual_channel_shuffle(x_res)

        # 空间注意力增强
        attn = self.spatial_attn(x_res)
        return x_res * (1 + attn)


class C2f_Star_RGB2(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_RGB2(self.c) for _ in range(n))


class C2f_Star_Depth2(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_Depth2(self.c) for _ in range(n))


class C2f_Star_Shuffle(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_Shuffle(self.c) for _ in range(n))


class C2f_Star_Enh(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Enhanced_Star_Block(self.c) for _ in range(n))


class C2f_Star_RGB3(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 增加纹理专用分支数量
        self.m = nn.ModuleList(Enhanced_Star_Block_RGB(self.c) for _ in range(n))


class Enhanced_Star_Block_RGB(Star_Block):
    def __init__(self, dim, mlp_ratio=4):  # 增加扩展率
        super().__init__(dim, mlp_ratio)

        # 多尺度纹理提取模块
        self.texture_extract = nn.ModuleList([
            nn.Sequential(
                Conv(dim, dim // 8, 1),
                Conv(dim // 8, dim // 8, 3),  # 常规尺度
                nn.ReLU6()
            ),
            nn.Sequential(
                Conv(dim, dim // 8, 1),
                Conv(dim // 8, dim // 8),  # 大尺度纹理
                nn.ReLU6()
            ),
            nn.Sequential(
                Conv(dim, dim // 8, 1),
                Conv(dim // 8, dim // 8, 1),  # 点状纹理
                # nn.AdaptiveAvgPool2d(1),
                nn.ReLU6()
            )
        ])

        # 纹理融合与增强
        self.texture_fusion = nn.Sequential(
            Conv(dim // 8 * 3, dim, 1),
            RGBChannelAttention(dim)  # 通道注意力聚焦重要纹理
        )

    def forward(self, x):
        # 原始特征提取
        base = super().forward(x)

        # 多尺度纹理提取
        texture_features = [branch(x) for branch in self.texture_extract]
        fused_texture = torch.cat(texture_features, dim=1)
        enhanced_texture = self.texture_fusion(fused_texture)

        return base + enhanced_texture  # 增强纹理特征


class RGBChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class C2f_Star_Depth3(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 增加几何特征分支
        self.m = nn.ModuleList(Enhanced_Star_Block_Depth(self.c) for _ in range(n))


class Enhanced_Star_Block_Depth(Star_Block):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__(dim, mlp_ratio)

        # 几何特征增强模块
        self.geometry_enhance = nn.Sequential(
            DepthSpatialAttention(),  # 空间注意力聚焦边缘
            Conv(dim, dim, 3, act=False),
            DeformableConv2d(dim, dim, kernel_size=3)  # 可变形卷积适应几何形状
        )

        # 边缘锐化分支
        self.edge_sharpen = nn.Sequential(
            Conv(dim, dim // 4, 1),
            Conv(dim // 4, dim // 4, 3),
            Conv(dim // 4, dim, 1, act=False),
            nn.Sigmoid()  # 生成边缘权重图
        )

    def forward(self, x):
        # 基础特征
        base = super().forward(x)

        # 几何增强
        geometry = self.geometry_enhance(x)

        # 边缘锐化
        edge_map = self.edge_sharpen(x)

        # 几何特征增强 + 边缘锐化
        return base + geometry * edge_map


class DepthSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class DeformableConv2d(nn.Module):
    """使用PyTorch官方DeformConv2d的可变形卷积实现"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # 每个点有2个偏移量(x,y)
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        # 生成偏移量
        offsets = self.offset_conv(x)
        # 应用可变形卷积
        return self.deform_conv(x, offsets)


class C3k2_RGB(C2f):
    """用于RGB分支的纹理特征提取"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 使用纹理优化的瓶颈块
        self.m = nn.ModuleList(
            TextureBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class TextureBottleneck(Bottleneck):
    """增强纹理提取的瓶颈块"""

    def __init__(self, c1, c2, shortcut=True, g=1):
        super().__init__(c1, c2, shortcut, g)
        # 增加纹理感知模块
        self.texture_aware = nn.Sequential(
            Conv(c2, c2 // 4, 1),
            nn.AvgPool2d(3, stride=1, padding=1),
            Conv(c2 // 4, c2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base = super().forward(x)
        texture_weights = self.texture_aware(x)
        return base * (1 + texture_weights)  # 增强纹理区域


class C3k2_Depth(C2f):
    """用于Depth分支的几何特征提取"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 使用几何优化的瓶颈块
        self.m = nn.ModuleList(
            GeometryBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class GeometryBottleneck(Bottleneck):
    """增强几何特征提取的瓶颈块"""

    def __init__(self, c1, c2, shortcut=True, g=1):
        super().__init__(c1, c2, shortcut, g)
        # 边缘增强分支
        self.edge_enhance = nn.Sequential(
            Conv(c1, c1 // 4, 3, act=False),
            nn.Conv2d(c1 // 4, c1 // 4, 3, padding=1, groups=c1 // 4),  # 深度卷积
            Conv(c1 // 4, c1, 1)
        )

    def forward(self, x):
        edge = self.edge_enhance(x)
        return super().forward(x) + edge


class DCGFM_Attn(nn.Module):
    def __init__(self, inc, reduction=8) -> None:
        """
        改进的带注意力机制的深度-彩色特征融合模块
        :param inc: 输入通道数列表 [rgb_channels, depth_channels]
        :param reduction: 通道注意力中的缩减比例
        """
        super().__init__()
        self.inc = inc

        # 第一层融合模块（带注意力）
        self.fusion1 = CGFM_Attn(inc, reduction)

        # 第二层融合模块（带注意力）
        self.fusion2 = CGFM_Attn([inc[1], inc[1]], reduction)  # 输入通道为[inc[1], inc[1]]

        # 跨模态交互模块
        self.cross_attn = CrossAttention(inc[1] * 2)

        # 空间注意力
        self.spatial_attn = SpatialAttention()

        # 通道注意力
        self.channel_attn = ChannelAttention(inc[1] * 2, reduction)

    def forward(self, x):
        x0, x1 = x  # RGB特征, Depth特征

        # 确保特征图尺寸一致
        if x0.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=x0.shape[2:], mode='bilinear', align_corners=False)

        # 第一层融合（带注意力）
        fused = self.fusion1((x0, x1))

        # 将融合结果拆分为两部分
        split_size = fused.size(1) // 2
        part1, part2 = torch.split(fused, split_size, dim=1)

        # 第二层融合（带注意力）
        final_fused = self.fusion2((part1, part2))

        # 跨模态交互
        cross_fused = self.cross_attn(final_fused)

        # 空间注意力
        spatial_attn = self.spatial_attn(cross_fused)
        attended = cross_fused * spatial_attn

        # 通道注意力
        channel_attn = self.channel_attn(attended)
        attended = attended * channel_attn

        return attended


class CGFM_Attn(nn.Module):
    """带注意力的彩色-几何特征融合模块"""

    def __init__(self, inc, reduction=8):
        """
        :param inc: 输入通道数列表 [rgb_channels, depth_channels]
        :param reduction: 通道注意力中的缩减比例
        """
        super().__init__()
        self.inc = inc

        # 特征转换层
        self.rgb_conv = Conv(inc[0], inc[1], 1)
        self.depth_conv = Conv(inc[1], inc[1], 1)

        # 自适应特征融合
        self.fusion_conv = Conv(inc[1] * 2, inc[1] * 2, 1)

        # 空间注意力
        self.spatial_attn = SpatialAttention()

        # 通道注意力
        self.channel_attn = ChannelAttention(inc[1] * 2, reduction)

        # 特征选择门控
        self.gate = nn.Sequential(
            Conv(inc[1] * 2, inc[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        rgb, depth = x

        # 确保特征图尺寸一致
        if rgb.shape[2:] != depth.shape[2:]:
            depth = F.interpolate(depth, size=rgb.shape[2:], mode='bilinear', align_corners=False)

        # 特征转换
        rgb_trans = self.rgb_conv(rgb)
        depth_trans = self.depth_conv(depth)

        # 特征融合
        fused = torch.cat([rgb_trans, depth_trans], dim=1)

        # 空间注意力
        spatial_attn = self.spatial_attn(fused)
        fused = fused * spatial_attn

        # 通道注意力
        channel_attn = self.channel_attn(fused)
        fused = fused * channel_attn

        # 特征选择门控
        gate_map = self.gate(fused)

        # 加权融合
        fused_rgb = rgb_trans * gate_map
        fused_depth = depth_trans * (1 - gate_map)

        return torch.cat([fused_rgb, fused_depth], dim=1)


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(combined)
        return self.sigmoid(attention)


class CrossAttention(nn.Module):
    """跨模态自注意力机制"""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim // 8, 1)
        self.key = nn.Conv2d(dim, dim // 8, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 生成query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        # 计算注意力权重
        attention = self.softmax(torch.bmm(query, key))

        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # 残差连接
        return self.gamma * out + x



