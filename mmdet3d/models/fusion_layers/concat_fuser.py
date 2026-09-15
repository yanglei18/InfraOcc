import torch
from torch import nn

from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class ConcatFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(
                in_channels * 2,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, camera_voxel_feat, lidar_voxel_feat):
        return self.fuse(torch.cat([camera_voxel_feat, lidar_voxel_feat], dim=1))
