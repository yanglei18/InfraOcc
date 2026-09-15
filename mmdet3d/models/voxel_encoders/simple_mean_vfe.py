import torch
import torch.nn as nn

from mmdet3d.models.builder import VOXEL_ENCODERS


@VOXEL_ENCODERS.register_module(name='RoadOccSimpleMeanVFE')
@VOXEL_ENCODERS.register_module()
class SimpleMeanVFE(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, features, num_points, coors, *args, **kwargs):
        features = features[:, :, :self.num_features].sum(dim=1)
        normalizer = torch.clamp(num_points.view(-1, 1).type_as(features), min=1.0)
        return features / normalizer
