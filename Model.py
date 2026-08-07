from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import numpy as np
from RGB_model import RegionRecoveryModel


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        # 调整weight和bias的维度以匹配输入形状 [B, C, H, W]
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)



def get_dynamic_kernel_sizes(C, min_k=3, max_k=9):
    """ 根据通道数 C 计算 kernel_sizes，确保是奇数 """
    base_k = int(np.log2(C) / 2)  # 计算基础核大小
    base_k = max(min_k, min(base_k, max_k))  # 限制范围
    return [k for k in [3, 5, 7, 9] if k <= base_k]  # 选出合适的 kernel_sizes



class ECAAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel_sizes = get_dynamic_kernel_sizes(channels)
        print(f"Using kernel_sizes={kernel_sizes} for channels={channels}")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, k, padding=(k - 1) // 2)
            for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()

        # 动态权重融合层
        if len(kernel_sizes) > 1:
            self.fusion = nn.Conv1d(len(kernel_sizes), len(kernel_sizes), kernel_size=1)
        else:
            self.fusion = None

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, 1, c)  # [B,1,C]

        # 多尺度卷积
        conv_outs = [conv(y) for conv in self.convs]  # 每个元素为[B,1,C]
        if len(conv_outs) > 1:
            combined = torch.cat(conv_outs, dim=1)  # [B,K,C]
            # 动态权重融合
            weights = self.fusion(combined)  # [B,K,C]
            weights = torch.softmax(weights, dim=1)
            fused = torch.sum(weights * combined, dim=1, keepdim=True)  # [B,1,C]
        else:
            fused = conv_outs[0]

        attn = self.sigmoid(fused).view(b, c, 1, 1)
        return x * attn.expand_as(x)
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# def Aggregate(dim, dim_out):
#     return nn.Sequential(
#         nn.Conv2d(dim, dim_out, 3, padding = 1),
#         LayerNorm(dim_out),
#         nn.MaxPool2d(3, stride = 2, padding = 1)
#     )
def Aggregate(dim, dim_out, use_eca=False):
    layers = [
        nn.Conv2d(dim, dim_out, 3, padding=1),  # 卷积降维
        LayerNorm(dim_out),  # 归一化
        nn.MaxPool2d(3, stride=2, padding=1)  # 下采样
    ]
    if use_eca:
        layers.append(ECAAttention(dim_out))
    return nn.Sequential(*layers)
def aAggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )
class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0., use_eca=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))
        self.use_eca = use_eca
        if use_eca:
            #self.eca = ECAAttention(channels=dim, kernel_sizes=[3])
            self.eca = ECAAttention(channels=dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            if self.use_eca:
                x = self.eca(x) + x  # 残差连接
                x = ff(x) + x


        return x

class HTNet(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        ause_eca=False,
        tuse_eca=False
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2 #
        fmap_size = image_size // patch_size #
        blocks = 2 ** (num_hierarchies - 1)#

        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout, use_eca=tuse_eca),
                Aggregate(dim_in, dim_out,use_eca=ause_eca) if not is_last else nn.Identity()
            ]))
        # for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
        #     is_last = level == 0
        #     depth = block_repeat
        #     self.layers.append(nn.ModuleList([
        #         Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
        #         Aggregate(dim_in, dim_out, use_eca=use_eca) if not is_last else nn.Identity()
        #     ]))


        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)
        #return self.mlp_head(x)
        return x

class CrossModalChannelFusion(nn.Module):
    def __init__(self, rgb_channels, flow_channels, reduction_ratio=4):
        super().__init__()

        # LayerNorm处理空间维度
        self.rgb_norm = nn.Sequential(
            nn.LayerNorm([rgb_channels, 1, 1]),
            #nn.Dropout(p=0.2)
        )
        self.flow_norm = nn.Sequential(
            nn.LayerNorm([rgb_channels, 1, 1]),
            #nn.Dropout(p=0.2)
        )
        # 通道转换模块
        self.rgb_adapter = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(rgb_channels // reduction_ratio, flow_channels, 1)
        )

        self.flow_adapter = nn.Sequential(
            nn.Conv2d(flow_channels, flow_channels // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(flow_channels // reduction_ratio, rgb_channels, 1)
        )

        # 调整输入通道为2*(rgb+flow)
        total_channels = 2 * (rgb_channels + flow_channels)
        self.cross_interaction = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(total_channels // reduction_ratio, total_channels, 1),
            nn.Sigmoid()
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(total_channels // reduction_ratio, total_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, flow):
        # 特征归一化
        rgb_norm = self.rgb_norm(rgb)
        flow_norm = self.flow_norm(flow)

        # 通道转换
        rgb_mapped = self.rgb_adapter(rgb_norm)  # R->F
        flow_mapped = self.flow_adapter(flow_norm)  # F->R

        # 拼接所有特征
        combined = torch.cat([rgb_norm, flow_norm, rgb_mapped, flow_mapped], dim=1)

        # 交互和注意力
        interaction_mask = self.cross_interaction(combined)
        channel_attn = self.channel_attn(combined)

        # 融合
        fused = combined * interaction_mask * channel_attn

        # 正确分割特征
        R, F = rgb.size(1), flow.size(1)
        fused_rgb = torch.cat([
            fused[:, :R],  # 原始RGB
            fused[:, 2 * R + F:]  # Flow转换到RGB
        ], dim=1)

        fused_flow = torch.cat([
            fused[:, R:R + F],  # 原始Flow
            fused[:, R + F:2 * R + F]  # RGB转换到Flow
        ], dim=1)

        return fused_rgb, fused_flow


class ModalityAwareFusion(nn.Module):
    def __init__(self, rgb_channels, flow_channels):
        super().__init__()
        self.cross_modal_fusion = CrossModalChannelFusion(rgb_channels, flow_channels)

        total_fused_channels = 2 * (rgb_channels + flow_channels)

        self.final_fusion = nn.Sequential(
            nn.Conv2d(total_fused_channels, 512, 1),  # 添加瓶颈结构
            LayerNorm(512),
            nn.GELU(),
            nn.Dropout2d(0.2),  # 增加Dropout
            nn.Conv2d(512, rgb_channels, 1)
        )
    def forward(self, rgb, flow):
        fused_rgb, fused_flow = self.cross_modal_fusion(rgb, flow)

        fused = torch.cat([fused_rgb, fused_flow], dim=1)
        return self.final_fusion(fused)
class FusionHTNet(nn.Module):
    def __init__(self, htnet_config, new_module_path):
        super().__init__()
        # 初始化原始HTNet
        self.htnet = HTNet(**htnet_config)
        # 初始化区域恢复模块
        self.region_recovery = RegionRecoveryModel()
        #self.alig = DFF(1024)
        self.norm = LayerNorm(1024)
        #self.dff = DFF_Enhanced(1024)
        self.fuse = ModalityAwareFusion(1024,1024)
        # 加载预训练权重
        if new_module_path:
            # state_dict = torch.load(new_module_path)
            # self.region_recovery.load_state_dict(state_dict, strict=True)
            try:
                # 1. 加载整个 checkpoint 字典
                checkpoint = torch.load(new_module_path, map_location='cuda')

                # 2. 从 checkpoint 字典中提取 'model_state_dict'
                model_state_for_recovery = checkpoint['model_state_dict']

                # 3. 将提取出的状态字典加载到 self.region_recovery 模块中
                self.region_recovery.load_state_dict(model_state_for_recovery, strict=True)

                print(f"成功从 '{new_module_path}' 加载 self.region_recovery 的权重")

                loaded_epoch = checkpoint['epoch']
                loaded_loss = checkpoint['loss']
                print(f"加载的 Epoch: {loaded_epoch}, Loss: {loaded_loss}")


            except KeyError:
                print(
                    f"错误：'{new_module_path}' 中缺少 'model_state_dict' 键。")
            except FileNotFoundError:
                print(f"错误：文件 '{new_module_path}' 不存在。")
            except Exception as e:
                print(f"加载权重时发生未知错误: {e}")


    def forward(self, x1, x2, x3):
        # HTNet特征提取
        ht_feat = self.htnet(x1)  # (b, c, h, w)
        #ht_feat = self.norm(ht_feat)
        # 区域恢复特征提取
        #with torch.no_grad():
        region_feat = self.region_recovery(x2, x3)
            #region_feat = self.norm(region_feat)
        fused = self.fuse(region_feat, ht_feat)
        #fused = ht_feat+region_feat

        return self.htnet.mlp_head(fused)
