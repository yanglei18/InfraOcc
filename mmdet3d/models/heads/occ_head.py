import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule

from mmdet3d.models.backbones.resnet import BasicBlock3D
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_classes=17,
            conv_before_predictor=True,
            class_weights=None,
            up_sample=True,
            last_occ_unsample=True,
            empty_idx=16,
            use_group_refine=False,
            group_list=None,
            foreground_idx=None,
            background_idx=None,
            class_frequency=None,
            bev_w_=200,
            bev_h_=200,
            bev_z_=16,
    ):
        super().__init__()
        if conv_before_predictor:
            self.voxel_conv = BasicBlock3D(
                channels_in=in_channels,
                channels_out=out_channels,
                stride=1,
            )

        self.predicter = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.Softplus(),
            nn.Linear(out_channels * 2, num_classes),
        )

        self.use_group_refine = use_group_refine
        self.empty_idx = empty_idx
        self.conv_before_predictor = conv_before_predictor
        self.last_occ_unsample = last_occ_unsample
        self.up_sample = up_sample
        self.foreground_idx = foreground_idx
        self.background_idx = background_idx
        self.class_frequency = class_frequency
        self.group_list = group_list
        self.bev_w_ = bev_w_
        self.bev_h_ = bev_h_
        self.bev_z_ = bev_z_
        self.num_classes = num_classes

    def forward(self, voxel_feats, last_occ_pred=None, **kwargs):
        if self.conv_before_predictor:
            pred_voxel_feats = self.voxel_conv(voxel_feats)
        else:
            pred_voxel_feats = voxel_feats

        if self.up_sample:
            pred_voxel_feats = F.interpolate(
                pred_voxel_feats,
                scale_factor=2,
                mode='trilinear',
                align_corners=False)

        pred_voxel_feats = pred_voxel_feats.permute(0, 4, 3, 2, 1)
        pred_voxel_semantic = self.predicter(pred_voxel_feats)

        return pred_voxel_semantic, pred_voxel_feats
