import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import (constant_init, kaiming_init, uniform_init)
import os
import cv2 as cv

CE = torch.nn.BCELoss(reduction='sum')
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable

"""
    多尺度特征融合模块,论文:https://arxiv.org/abs/2009.14082
"""
__all__ = (
    "DAF",
    "iAFF",
    "AFF",
    "MS_CAM",
    "Add",
    "Mul",
    "Add2",
    "GPT",
    "myTransformerBlock",
    "SelfAttention",
    "SqueezeAndExcitation",
    "Excitation",
    "SqueezeAndExciteFusionAdd",
    "ExciteFusionAdd",
    "ResidualExciteFusion",
    "ViTFlattener",
    "ViTUnFlattener",
    "SelfAttentionFusion",
    "ResidualAttentionFusion",
    "MHAttentionFusionSecond",
    "MHAttentionFusionThird",
    "CrossModalMultiHeadAttention",
    "Concat3",
    "ChannelAttention",
    "SpatialAttention",
    "Mutual_info_reg",
    "PAdd",
)


class DAF(nn.Module):
    '''
    直接相加 DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xa = x[0] + x[1]
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x[0] * wei + x[1] * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x[0] * wei2 + x[1] * (1 - wei2)
        return xo


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


"""
参考论文：https://arxiv.org/abs/2111.00273
"""


class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        # 加法
        return torch.add(x[0], x[1])
        # 乘法
        # return torch.mul(x[0], x[1])





class Mul(nn.Module):
    #  Mul two tensors
    def __init__(self, arg):
        super(Mul, self).__init__()
        self.arg = arg

    def forward(self, x):
        # 乘法
        return torch.mul(x[0], x[1])


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

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
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


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


"""
参考论文：https://arxiv.org/pdf/2011.06961
"""


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        if rgb.sum().item() < 1e-6:
            pass
        else:
            rgb = self.se_rgb(rgb)

        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        out = rgb + depth
        return out


class ExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ExciteFusionAdd, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                 activation=activation)
        self.se_depth = Excitation(channels_in,
                                   activation=activation)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        if rgb.sum().item() < 1e-6:
            pass
        else:
            rgb = self.se_rgb(rgb)

        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        out = rgb + depth
        return out


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


class ViTFlattener(nn.Module):

    def __init__(self, patch_dim):
        super(ViTFlattener, self).__init__()
        self.patch_dim = patch_dim
        self.patcher = torch.nn.PixelUnshuffle(self.patch_dim)
        self.flattener = torch.nn.Flatten(-2, -1)

    def forward(self, inp):
        patches = self.patcher(inp)
        flat = self.flattener(patches)
        ViT_out = flat
        return ViT_out


class ViTUnFlattener(nn.Module):

    def __init__(self, patch_dim):
        super(ViTUnFlattener, self).__init__()
        self.patch_dim = patch_dim
        self.unpatcher = torch.nn.PixelShuffle(self.patch_dim)

    def forward(self, inp, out_shape):
        _, C, H, W = out_shape
        x = inp
        x = x.reshape(-1, C * self.patch_dim * self.patch_dim, H // self.patch_dim, W // self.patch_dim)
        x = self.unpatcher(x)
        return x


class SelfAttentionFusion(nn.Module):
    def __init__(self, patches_size, channels, bottleneck_dim=32):
        super(SelfAttentionFusion, self).__init__()

        self.patches_size = patches_size
        self.bottleneck_dim = bottleneck_dim
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim

        self.downsampler_key_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                           stride=1)
        self.downsampler_query_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                             stride=1)
        self.downsampler_value_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                             stride=1)
        self.downsampler_key_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                           stride=1)
        self.downsampler_query_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                             stride=1)
        self.downsampler_value_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                             stride=1)
        self.vit_flatten = ViTFlattener(self.patches_size)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)
        self.upsampler_1 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)
        self.upsampler_2 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        # Self-Attention for RGB
        query_rgb = self.downsampler_query_1(rgb)
        key_rgb = self.downsampler_key_1(rgb)
        value_rgb = self.downsampler_value_1(rgb)
        flattened_query_rgb = self.vit_flatten(query_rgb)
        flattened_key_rgb = self.vit_flatten(key_rgb)
        flattened_value_rgb = self.vit_flatten(value_rgb)

        QKt_rgb = torch.matmul(flattened_query_rgb, flattened_key_rgb.permute(0, 2, 1)) / self.scale
        attention_weight_rgb = self.softmax(QKt_rgb)
        output_rgb = torch.matmul(attention_weight_rgb, flattened_value_rgb)
        output_rgb = self.vit_unflatten(output_rgb, query_rgb.shape)
        output_rgb = self.upsampler_1(output_rgb)

        # Self-Attention for Depth
        query_depth = self.downsampler_query_2(depth)
        key_depth = self.downsampler_key_2(depth)
        value_depth = self.downsampler_value_2(depth)
        flattened_query_depth = self.vit_flatten(query_depth)
        flattened_key_depth = self.vit_flatten(key_depth)
        flattened_value_depth = self.vit_flatten(value_depth)

        QKt_depth = torch.matmul(flattened_query_depth, flattened_key_depth.permute(0, 2, 1)) / self.scale
        attention_weight_depth = self.softmax(QKt_depth)
        output_depth = torch.matmul(attention_weight_depth, flattened_value_depth)
        output_depth = self.vit_unflatten(output_depth, query_depth.shape)
        output_depth = self.upsampler_2(output_depth)

        # Merging
        output = output_rgb + output_depth
        return output


class ResidualAttentionFusion(nn.Module):
    def __init__(self, patches_size, channels, alpha=1., bottleneck_dim=32):
        super(ResidualAttentionFusion, self).__init__()

        self.alpha = alpha

        self.patches_size = patches_size
        self.bottleneck_dim = bottleneck_dim
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim

        self.downsampler_key_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                           stride=1)
        self.downsampler_query_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                             stride=1)
        self.downsampler_value_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                             stride=1)
        self.downsampler_key_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                           stride=1)
        self.downsampler_query_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                             stride=1)
        self.downsampler_value_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                             stride=1)
        self.vit_flatten = ViTFlattener(self.patches_size)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)
        self.upsampler_1 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)
        self.upsampler_2 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        # Self-Attention for RGB
        query_rgb = self.downsampler_query_1(rgb)
        key_rgb = self.downsampler_key_1(rgb)
        value_rgb = self.downsampler_value_1(rgb)
        flattened_query_rgb = self.vit_flatten(query_rgb)
        flattened_key_rgb = self.vit_flatten(key_rgb)
        flattened_value_rgb = self.vit_flatten(value_rgb)

        QKt_rgb = torch.matmul(flattened_query_rgb, flattened_key_rgb.permute(0, 2, 1)) / self.scale
        attention_weight_rgb = self.softmax(QKt_rgb)
        output_rgb = torch.matmul(attention_weight_rgb, flattened_value_rgb)
        output_rgb = self.vit_unflatten(output_rgb, query_rgb.shape)
        output_rgb = self.upsampler_1(output_rgb)

        # Self-Attention for Depth
        query_depth = self.downsampler_query_2(depth)
        key_depth = self.downsampler_key_2(depth)
        value_depth = self.downsampler_value_2(depth)
        flattened_query_depth = self.vit_flatten(query_depth)
        flattened_key_depth = self.vit_flatten(key_depth)
        flattened_value_depth = self.vit_flatten(value_depth)

        QKt_depth = torch.matmul(flattened_query_depth, flattened_key_depth.permute(0, 2, 1)) / self.scale
        attention_weight_depth = self.softmax(QKt_depth)
        output_depth = torch.matmul(attention_weight_depth, flattened_value_depth)
        output_depth = self.vit_unflatten(output_depth, query_depth.shape)
        output_depth = self.upsampler_2(output_depth)

        # Merging
        output = rgb + self.alpha * (output_rgb + output_depth)
        return output


class MHAttentionFusionSecond(nn.Module):
    def __init__(self, patches_size, channels, bottleneck_dim=32):
        super(MHAttentionFusionSecond, self).__init__()

        self.patches_size = patches_size
        self.bottleneck_dim = channels
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim

        self.vit_flatten = ViTFlattener(self.patches_size)
        self.linear_rgb_q = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_rgb_k = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_rgb_v = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_depth_q = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_depth_k = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_depth_v = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.linear_rgb = nn.Linear(in_features=bottleneck_dim, out_features=self.latent_patch_dim)
        self.linear_depth = nn.Linear(in_features=bottleneck_dim, out_features=self.latent_patch_dim)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        # Self-Attention for RGB
        vit_rgb = self.vit_flatten(rgb)
        q_rgb = self.linear_rgb_q(vit_rgb)
        k_rgb = self.linear_rgb_k(vit_rgb)
        v_rgb = self.linear_rgb_v(vit_rgb)
        QKt_rgb = torch.matmul(q_rgb, k_rgb.permute(0, 2, 1)) / self.scale
        attention_weight_rgb = self.softmax(QKt_rgb)
        output_rgb = torch.matmul(attention_weight_rgb, v_rgb)
        output_rgb = self.linear_rgb(output_rgb)
        output_rgb = self.vit_unflatten(output_rgb, rgb.shape)

        # Self-Attention for Depth
        vit_depth = self.vit_flatten(depth)
        q_depth = self.linear_depth_q(vit_depth)
        k_depth = self.linear_depth_k(vit_depth)
        v_depth = self.linear_depth_v(vit_depth)
        QKt_depth = torch.matmul(q_depth, k_depth.permute(0, 2, 1)) / self.scale
        attention_weight_depth = self.softmax(QKt_depth)
        output_depth = torch.matmul(attention_weight_depth, v_depth)
        output_depth = self.linear_depth(output_depth)
        output_depth = self.vit_unflatten(output_depth, depth.shape)

        # Merging
        output = output_rgb + output_depth
        return output


class MHAttentionFusionThird(nn.Module):
    def __init__(self, patches_size, channels, bottleneck_dim=32):
        super(MHAttentionFusionThird, self).__init__()

        self.patches_size = patches_size
        self.bottleneck_dim = bottleneck_dim
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim

        self.downsampler_key = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                         stride=1)
        self.downsampler_query = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                           stride=1)
        self.downsampler_value = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1,
                                           stride=1)
        self.vit_flatten = ViTFlattener(self.patches_size)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)
        self.upsampler = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)

    def forward(self, x):
        rgb, depth = x[0], x[1]
        # Cross-Attention
        query = self.downsampler_query(depth)
        key = self.downsampler_key(rgb)
        value = self.downsampler_value(depth)

        flattened_query = self.vit_flatten(query)
        flattened_key = self.vit_flatten(key)
        flattened_value = self.vit_flatten(value)

        QKt = torch.matmul(flattened_query, flattened_key.permute(0, 2, 1)) / self.scale
        attention_weight = self.softmax(QKt)
        output = torch.matmul(attention_weight, flattened_value)
        output = self.vit_unflatten(output, query.shape)
        output = self.upsampler(output)
        return output


class CrossModalMultiHeadAttention(nn.Module):
    '''
    来源于论文：RGB-D Grasp Detection via Depth Guided Learning with Cross-modal Attention
    '''

    def __init__(self,
                 in_channels,
                 num_head,
                 ratio):
        super(CrossModalMultiHeadAttention, self).__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.out_channels = int(in_channels * ratio)
        self.query_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
        self.key_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
        self.value_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
        self.W = nn.Conv2d(in_channels=self.out_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.fuse = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels, in_channels, kernel_size=1)
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1, bias=False)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x, img_metas=None):
        key = x[0]
        query = x[1]
        # key:RGB; query:Depth
        batch, channels, height, width = query.size()
        q_out = self.query_conv(query).contiguous().view(batch, self.num_head, -1, height, width)
        k_out = self.key_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
        v_out = self.value_conv(query).contiguous().view(batch, self.num_head, -1, height, width)

        att = (q_out * k_out).sum(dim=2) / np.sqrt(self.out_channels // self.num_head)

        if self.num_head == 1:
            softmax = att.unsqueeze(dim=2)
            # softmax = torch.sigmoid(att).unsqueeze(dim=2)
        else:
            # softmax = F.softmax(att, dim=1).unsqueeze(dim=2)
            softmax = torch.sigmoid(att).unsqueeze(dim=2)

        weighted_value = v_out * softmax
        # weighted_value = weighted_value.sum(dim=1)
        weighted_value = weighted_value.view(batch, self.out_channels, height, width)
        out = query + self.W(weighted_value)
        # out = self.W(weighted_value)
        out = self.bn(out)

        debug = False
        if debug and img_metas is not None:
            dir = os.path.join('/home/qinran_2020/mmdetection_grasp/eval/cross_attention_sigmoid/attention')
            scene_dir = os.path.join(dir, 'scene_%04d' % img_metas[0]['sceneId'])
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            img_name = os.path.join(scene_dir, '%04d' % img_metas[0]['annId'])

            img = torch.sigmoid(att)[0][0].detach().cpu().numpy().astype(np.float32)
            # img[img > 0.7] = 0.5
            # img[img < -1] = 0
            # img = np.mean(img, axis=0)
            heatmap = None
            heatmap = cv.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
            cv.imwrite(img_name + "_attention.png", heatmap)

            # channels = len(out[0])
            # for i in range(channels):
            #     img = out[0][i].detach().cpu().numpy().astype(np.float32)
            #     heatmap = None
            #     heatmap = cv.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            #     heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
            #     cv.imwrite(img_name + "_%d.png" % i, heatmap)

        return self.fuse(torch.cat([key, out], dim=1))


"""
SLBAF-Net
"""


class Concat3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2, ratio=16, kernel_size=7, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.spatial_attention = SpatialAttention(7)
        self.channel_attention = ChannelAttention(c1, ratio)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        weight1 = self.spatial_attention(x1)
        weight2 = self.spatial_attention(x2)
        weight = (weight1 / weight2)
        x2 = weight * x2
        x1 = x1 * (2 - weight)
        x = torch.cat((x1, x2), self.d)
        x = x * self.channel_attention(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

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


"""
论文:
RGB-D Saliency Detection via Cascaded Mutual Information Minimization
"""


class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        self.fc1_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc2_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc1_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc2_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)

        self.fc1_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        self.fc2_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        self.fc1_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        self.fc2_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)

        self.fc1_rgb3 = nn.Linear(channels * 1 * 28 * 28, latent_size)
        self.fc2_rgb3 = nn.Linear(channels * 1 * 28 * 28, latent_size)
        self.fc1_depth3 = nn.Linear(channels * 1 * 28 * 28, latent_size)
        self.fc2_depth3 = nn.Linear(channels * 1 * 28 * 28, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.bn1(self.layer1(rgb_feat))))
        depth_feat = self.layer4(self.leakyrelu(self.bn2(self.layer2(depth_feat))))
        # print(rgb_feat.size())
        # print(depth_feat.size())
        if rgb_feat.shape[2] == 16:
            rgb_feat = rgb_feat.view(-1, self.channel * 1 * 16 * 16)
            depth_feat = depth_feat.view(-1, self.channel * 1 * 16 * 16)

            mu_rgb = self.fc1_rgb1(rgb_feat)
            logvar_rgb = self.fc2_rgb1(rgb_feat)
            mu_depth = self.fc1_depth1(depth_feat)
            logvar_depth = self.fc2_depth1(depth_feat)
        elif rgb_feat.shape[2] == 22:
            rgb_feat = rgb_feat.view(-1, self.channel * 1 * 22 * 22)
            depth_feat = depth_feat.view(-1, self.channel * 1 * 22 * 22)
            mu_rgb = self.fc1_rgb2(rgb_feat)
            logvar_rgb = self.fc2_rgb2(rgb_feat)
            mu_depth = self.fc1_depth2(depth_feat)
            logvar_depth = self.fc2_depth2(depth_feat)
        else:
            rgb_feat = rgb_feat.view(-1, self.channel * 1 * 28 * 28)
            depth_feat = depth_feat.view(-1, self.channel * 1 * 28 * 28)
            mu_rgb = self.fc1_rgb3(rgb_feat)
            logvar_rgb = self.fc2_rgb3(rgb_feat)
            mu_depth = self.fc1_depth3(depth_feat)
            logvar_depth = self.fc2_depth3(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm, z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth + ce_depth_rgb - bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss, z_rgb, z_depth


# parameter与add结合
class PAdd(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(PAdd, self).__init__()
        self.arg = arg
        self.rgb = Parameter(torch.ones(1))
        self.depth = Parameter(torch.ones(1))

    def forward(self, x):
        # 加法
        return torch.add(self.rgb * x[0], self.depth * x[1])
        # 乘法
        # return torch.mul(x[0], x[1])
