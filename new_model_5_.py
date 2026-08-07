import math

import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import cv2
from einops import rearrange
import numpy as np
import os
from torch.utils.checkpoint import checkpoint

class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)

        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OptimDualBranchBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.expansion = 4  # 通道扩展系数

        # 低分辨率分支
        self.low_res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels, in_channels * self.expansion, 1),
            nn.SiLU(),
            ChannelAttention(in_channels * self.expansion),
            nn.Conv2d(in_channels * self.expansion, in_channels * self.expansion, 1),
            nn.PixelShuffle(2)
        )

        # 高分辨率分支
        self.high_res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(0.1),
            ChannelAttention(in_channels),
            nn.Conv2d(in_channels, in_channels, 1)
        )

        # 融合层
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            DyT(in_channels),
            nn.GELU()
        )

        # 下采样层
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, stride=2, padding=1),
            DyT(in_channels * 2),
            nn.GELU()
        )

    def forward(self, x):
        low = self.low_res(x)
        high = self.high_res(x)
        fused = self.fuse_conv(torch.cat([low, high], dim=1))
        fused = fused + x
        return self.downsample(fused)



class Encoder(nn.Module):
    #双分支编码器

    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # 初始下采样层（224→112）
            nn.Conv2d(in_channels, base_dim, 7, stride=2, padding=3),
            DyT(base_dim),
            nn.GELU(),

            OptimDualBranchBlock(base_dim),  # 输出56x56（通道128）
            OptimDualBranchBlock(base_dim * 2),  # 输出28x28（通道256）
            OptimDualBranchBlock(base_dim * 4),  # 输出14x14（通道512）
        )

    def forward(self, x):
        return self.net(x)


# model.py


class EfficientCrossAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8, expansion_ratio=2):
        super().__init__()
        self.attn_dropout = nn.Dropout(0.1)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expansion = expansion_ratio

        # 分开投影层：
        # mix_feat 提取 q
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        # orig_feat 提取 k
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        # orig_feat 提取 v（扩展通道）
        self.v_proj = nn.Conv2d(dim, dim * expansion_ratio, 1)

        # 动态门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        # 轻量位置编码
        self.pos_enc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        # 混合FFN增强
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            ChannelAttention(dim * 2),
            nn.Conv2d(dim * 2, dim, 1)
        )

        # 输出投影层，将扩展后的通道数降回原始通道数
        self.out_proj = nn.Conv2d(dim * expansion_ratio, dim, 1)

    def scaled_dot_product_attention(self, q, k, v):
        """scaled dot-product attention"""
        d_k = q.shape[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def forward(self, orig_feat, mix_feat):
        B, C, H, W = orig_feat.shape

        # 分别提取 q, k, v：
        q = self.q_proj(mix_feat)         # q 来自 mix_feat
        k = self.k_proj(orig_feat)          # k 来自 orig_feat
        v = self.v_proj(orig_feat)          # v 来自 orig_feat
        v = F.gelu(v)

        # 添加位置编码
        q = q + self.pos_enc(q)
        k = k + self.pos_enc(k)
        # 将v的通道拆分为多个dim通道组
        v_groups = v.chunk(self.expansion, dim=1)  # 按扩展倍数分组

        # 对每个分组应用预训练的位置编码
        v_encoded = []
        for group in v_groups:
            aligned_group = group[:, :self.dim, :, :]  # 取前dim通道
            encoded_group = self.pos_enc(aligned_group)
            v_encoded.append(encoded_group)

        # 合并结果
        v = v + torch.cat(v_encoded, dim=1)

        # 多头拆分
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        v = rearrange(v, 'b (h e d) x y -> b h (x y) (e d)', h=self.num_heads, e=self.expansion)

        out = self.scaled_dot_product_attention(q, k, v)

        # 合并头部，out 形状变为 (B, dim*expansion, H*W)
        out = rearrange(out, 'b h l d -> b (h d) l')
        out = out.view(B, self.dim * self.expansion, H, W)

        # 输出投影，将通道数降回原始 dim
        out = self.out_proj(out)

        # 动态门控融合
        gate = self.gate(orig_feat)
        return orig_feat + gate * out + self.ffn(out)


class CrossRegionAttention(nn.Module):
    def __init__(self, dim=512, depth=2):
        super().__init__()
        self.layers = nn.ModuleList([
            EfficientCrossAttention(dim)
            for _ in range(depth)
        ])

    def forward(self, orig_feat, mix_feat):
        for layer in self.layers:
            mix_feat = layer(orig_feat, mix_feat)
        return mix_feat

class Decoder(nn.Module):
    def __init__(self, in_dim=512, out_channels=3):  # 输入维度与Encoder输出一致
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, 3, 2, 1, output_padding=1),  # 14→28
            DyT(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),  # 28→56
            DyT(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),  # 56→112
            DyT(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),  # 112→224
            DyT(32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Tanh()
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class RegionRecoveryModel(nn.Module):
    """完整恢复模型"""

    def __init__(self):
        super().__init__()
        self.global_dropout = nn.Dropout2d(0.1)
        self.encoder_orig = Encoder()
        self.encoder_mix = Encoder()
        self.attention = CrossRegionAttention()
        self.decoder = Decoder()
        self.conv = nn.Conv2d(512, 1024, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    #主模型1
    def forward(self, orig_img,mixed_img, is_pretrain=False):
        orig_feat = self.encoder_orig(orig_img)
        mix_feat = self.encoder_mix(mixed_img)

        orig_feat = self.global_dropout(orig_feat)
        mix_feat = self.global_dropout(mix_feat)

        # 尺寸验证
        assert orig_feat.shape[-2:] == mix_feat.shape[-2:], \
            f"特征图尺寸不匹配: orig {orig_feat.shape} vs mix {mix_feat.shape}"

        attn_feat = self.attention(orig_feat, mix_feat)
        # print(self.decoder(attn_feat).shape)
        if is_pretrain:
            return self.decoder(attn_feat)
        else:
            attn_feat = self.conv(attn_feat)
            attn_feat = self.pool(attn_feat)
            return attn_feat


if __name__ == "__main__":
    # encoder = Encoder()

    x1 = torch.randn(64, 3, 224, 224)  # 输入图像
    x2 = torch.randn(64, 3, 224, 224)  # 输入图像
    attention = RegionRecoveryModel()
    attn_out = attention(x1,x2)
