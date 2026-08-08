import torch
from torch import nn

from .htnet import HTNet, LayerNorm
from .rgb_branch import RegionRecoveryModel

class ECAWeight(nn.Module):
    """
    Efficient Channel Attention.
    Returns channel weights with shape [B, C, 1, 1].
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size - 1) // 2,bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [B,C,H,W] -> [B,C,1,1]
        y = self.avg_pool(x)

        # [B,C,1,1] -> [B,1,C]
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)
        y = self.sigmoid(y)

        # [B,1,C] -> [B,C,1,1]
        y = y.transpose(-1, -2).unsqueeze(-1)

        return y
class CrossModalChannelFusion(nn.Module):
    def __init__(
        self,
        rgb_channels,
        flow_channels,
        reduction_ratio=4
    ):
        super().__init__()

        # Spatial-size-independent channel normalization
        self.rgb_norm = LayerNorm(rgb_channels)
        self.flow_norm = LayerNorm(flow_channels)

        # RGB -> Flow
        self.rgb_adapter = nn.Sequential(
            nn.Conv2d(rgb_channels,rgb_channels // reduction_ratio,1),
            nn.GELU(),
            nn.Conv2d(rgb_channels // reduction_ratio,flow_channels,1)
        )

        # Flow -> RGB
        self.flow_adapter = nn.Sequential(
            nn.Conv2d(flow_channels,flow_channels // reduction_ratio,1),
            nn.GELU(),
            nn.Conv2d(flow_channels // reduction_ratio,rgb_channels,1)
        )

        # Channel order:
        # [RGB | Flow | RGB->Flow | Flow->RGB]
        total_channels = 2 * (
            rgb_channels + flow_channels
        )

        # Spatial interaction:
        # [B, C, H, W] -> [B, 1, H, W]
        self.spatial_interaction = nn.Sequential(
            # Channel reduction
            nn.Conv2d(total_channels,total_channels // reduction_ratio,kernel_size=1),
            nn.GELU(),

            # Spatial interaction
            nn.Conv2d(total_channels // reduction_ratio,1,kernel_size=3,padding=1),
            nn.Sigmoid()
        )

        # Efficient Channel Attention
        # [B, C, H, W] -> [B, C, 1, 1]
        self.channel_attn = ECAWeight(total_channels,kernel_size=3)

    def forward(self, rgb, flow):

        # ---------------------------------------------------
        # 1. Normalization
        # ---------------------------------------------------
        rgb_norm = self.rgb_norm(rgb)
        flow_norm = self.flow_norm(flow)

        # ---------------------------------------------------
        # 2. Reciprocal modality adaptation
        # ---------------------------------------------------
        rgb_mapped = self.rgb_adapter(rgb_norm)    # RGB -> Flow
        flow_mapped = self.flow_adapter(flow_norm) # Flow -> RGB

        # ---------------------------------------------------
        # 3. Joint representation
        #
        # channel order:
        # [ RGB | Flow | RGB->Flow | Flow->RGB ]
        #    R      F        F           R
        # ---------------------------------------------------
        combined = torch.cat([rgb_norm,flow_norm,rgb_mapped,flow_mapped],dim=1)

        # ---------------------------------------------------
        # 4. Spatial and channel weighting
        # ---------------------------------------------------
        spatial_mask = self.spatial_interaction(combined)

        channel_weight = self.channel_attn(combined)

        fused = (combined* spatial_mask * channel_weight)

        # ---------------------------------------------------
        # 5. Split into two enhanced modality paths
        # ---------------------------------------------------
        R = rgb.size(1)
        F = flow.size(1)

        rgb_start = 0
        flow_start = R
        rgb_to_flow_start = R + F
        flow_to_rgb_start = R + 2 * F

        # RGB + Flow->RGB
        fused_rgb = torch.cat([fused[:,rgb_start:rgb_start + R],fused[:,flow_to_rgb_start:flow_to_rgb_start + R]],dim=1)

        # Flow + RGB->Flow
        fused_flow = torch.cat([fused[:,flow_start:flow_start + F],fused[:,rgb_to_flow_start:rgb_to_flow_start + F]],dim=1)

        return fused_rgb, fused_flow


class ModalityAwareFusion(nn.Module):
    def __init__(self,rgb_channels,flow_channels):
        super().__init__()
        self.cross_modal_fusion = CrossModalChannelFusion(rgb_channels,flow_channels)

        # fused_rgb = 2R
        # fused_flow = 2F
        total_channels = 2 * (rgb_channels + flow_channels)

        self.final_fusion = nn.Sequential(
            nn.Conv2d(
                total_channels,
                512,
                kernel_size=1
            ),
            LayerNorm(512),
            nn.GELU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(512,rgb_channels,kernel_size=1)
        )

    def forward(self, rgb, flow):

        fused_rgb, fused_flow = self.cross_modal_fusion(rgb,flow)
        # ReConcat
        fused = torch.cat([fused_rgb,fused_flow],dim=1)

        return self.final_fusion(fused)
class FusionHTNet(nn.Module):
    def __init__(self, htnet_config, new_module_path):
        super().__init__()

        self.htnet = HTNet(**htnet_config)

        self.region_recovery = RegionRecoveryModel()

        # HTNet first hierarchy:
        # 256 channels -> 1024 channels
        self.flow_spatial_proj = nn.Sequential(
            nn.Conv2d(htnet_config["dim"],1024,kernel_size=1),
            LayerNorm(1024),
            nn.GELU()
        )

        self.fuse = ModalityAwareFusion(1024,1024)

        if new_module_path:
            checkpoint = torch.load(
                new_module_path,
                map_location="cpu"
            )

            if "model_state_dict" not in checkpoint:
                raise KeyError(
                    f"'model_state_dict' not found in checkpoint: "
                    f"{new_module_path}"
                )

            self.region_recovery.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True
            )

            print(
                f"Successfully loaded pretrained weights from: "
                f"{new_module_path}"
            )

    def forward(self, x1, x2, x3):

        # ---------------------------------------------------
        # Flow branch
        # ---------------------------------------------------
        flow_global, flow_spatial = self.htnet(x1,return_spatial=True)

        # [B,256,4,4] -> [B,1024,4,4]
        flow_spatial = self.flow_spatial_proj(flow_spatial)

        # [B,1024,1,1] -> [B,1024,4,4]
        flow_global = torch.nn.functional.interpolate(flow_global,size=flow_spatial.shape[-2:],mode="nearest")
        flow_feat = (flow_spatial+ flow_global)

        # ---------------------------------------------------
        # RGB branch
        # IRSA output:
        # [B,1024,14,14]
        # ---------------------------------------------------
        region_feat = self.region_recovery(x3,x2,return_spatial=True)

        # Align RGB spatial resolution with Flow
        region_feat = torch.nn.functional.adaptive_avg_pool2d(
            region_feat,
            flow_feat.shape[-2:]
        )

        # ---------------------------------------------------
        # MAFM
        # ---------------------------------------------------
        fused = self.fuse(region_feat,flow_feat)

        # Existing HTNet classification head can still be used.
        return self.htnet.mlp_head(fused)