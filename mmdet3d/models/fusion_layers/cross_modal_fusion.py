import torch
from torch import nn

from mmcv.cnn import ConvModule

from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class Cross_Modal_Fusion(nn.Module):
    def __init__(self,
                 kernel_size=3,
                 in_channels=None,
                 img_channels=None,
                 lidar_channels=None,
                 out_channels=256,
                 norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01)):
        super().__init__()

        if kernel_size not in (3, 7):
            raise AssertionError('kernel size must be 3 or 7')

        if in_channels is not None:
            img_channels = in_channels if img_channels is None else img_channels
            lidar_channels = in_channels if lidar_channels is None else lidar_channels
        if img_channels is None or lidar_channels is None:
            raise ValueError('Cross_Modal_Fusion requires img_channels and lidar_channels.')

        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )
        self.att_lidar = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )
        self.reduce_mix_voxel = ConvModule(
            img_channels + lidar_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            inplace=False,
        )

    @staticmethod
    def _channel_attention(feat, att_layer):
        feat_avg = torch.mean(feat, dim=1, keepdim=True)
        feat_max, _ = torch.max(feat, dim=1, keepdim=True)
        feat_avg_max = torch.cat([feat_avg, feat_max], dim=1)
        return att_layer(feat_avg_max)

    def forward(self, img_voxel_feat, lidar_voxel_feat):
        img_att = self._channel_attention(img_voxel_feat, self.att_img)
        lidar_att = self._channel_attention(lidar_voxel_feat, self.att_lidar)

        img_voxel_feat = img_voxel_feat * lidar_att
        lidar_voxel_feat = lidar_voxel_feat * img_att
        fusion_voxel = torch.cat([img_voxel_feat, lidar_voxel_feat], dim=1)
        return self.reduce_mix_voxel(fusion_voxel)
