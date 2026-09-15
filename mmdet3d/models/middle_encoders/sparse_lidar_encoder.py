from mmcv.cnn import build_norm_layer
import torch.nn as nn

import spconv.pytorch as spconv

from mmdet3d.models.builder import MIDDLE_ENCODERS


def _post_act_block(in_channels,
                    out_channels,
                    kernel_size,
                    indice_key=None,
                    stride=1,
                    padding=1,
                    conv_type='subm',
                    norm_cfg=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
            indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            indice_key=indice_key)
    else:
        raise NotImplementedError(conv_type)

    return spconv.SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(inplace=True),
    )


class _SparseBasicBlock(spconv.SparseModule):
    def __init__(self, inplanes, planes, norm_cfg=None, indice_key=None):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(
                inplanes,
                planes,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        out = out.replace_feature(out.features + x.features)
        out = out.replace_feature(self.relu(out.features))
        return out


@MIDDLE_ENCODERS.register_module()
class SparseLiDARVoxelEncoder(nn.Module):
    def __init__(self,
                 input_channel,
                 base_channel,
                 out_channel,
                 sparse_shape_xyz,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 **kwargs):
        super().__init__()
        self.sparse_shape_xyz = sparse_shape_xyz
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel, base_channel, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, base_channel)[1],
            nn.ReLU(inplace=True),
        )
        self.conv_body = spconv.SparseSequential(
            _SparseBasicBlock(
                base_channel, base_channel, norm_cfg=norm_cfg, indice_key='roadocc_res'),
            _SparseBasicBlock(
                base_channel, base_channel, norm_cfg=norm_cfg, indice_key='roadocc_res'),
        )
        self.conv_out = _post_act_block(
            base_channel,
            out_channel,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            indice_key='roadocc_out')

    def forward(self, voxel_features, coors, batch_size):
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features,
            coors.int(),
            self.sparse_shape_xyz[::-1],
            batch_size)
        x = self.conv_input(input_sp_tensor)
        x = self.conv_body(x)
        x = self.conv_out(x)
        return x.dense().permute(0, 1, 4, 3, 2).contiguous()


@MIDDLE_ENCODERS.register_module()
class SparseLiDAREnc8x(nn.Module):
    """OpenOcc-style sparse LiDAR encoder with 8x spatial downsampling.

    This keeps the downstream interface compatible with the existing RoadOcc
    and V2XOcc decoders by returning the dense tensor under the ``x`` key.
    """

    def __init__(self,
                 input_channel,
                 base_channel,
                 out_channel,
                 sparse_shape_xyz,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 **kwargs):
        super().__init__()
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            _post_act_block(
                base_channel,
                base_channel * 2,
                3,
                stride=2,
                padding=1,
                indice_key='spconv1',
                conv_type='spconv',
                norm_cfg=norm_cfg),
            _SparseBasicBlock(
                base_channel * 2,
                base_channel * 2,
                norm_cfg=norm_cfg,
                indice_key='res1'),
            _SparseBasicBlock(
                base_channel * 2,
                base_channel * 2,
                norm_cfg=norm_cfg,
                indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            _post_act_block(
                base_channel * 2,
                base_channel * 4,
                3,
                stride=2,
                padding=1,
                indice_key='spconv2',
                conv_type='spconv',
                norm_cfg=norm_cfg),
            _SparseBasicBlock(
                base_channel * 4,
                base_channel * 4,
                norm_cfg=norm_cfg,
                indice_key='res2'),
            _SparseBasicBlock(
                base_channel * 4,
                base_channel * 4,
                norm_cfg=norm_cfg,
                indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            _post_act_block(
                base_channel * 4,
                base_channel * 8,
                3,
                stride=2,
                padding=1,
                indice_key='spconv3',
                conv_type='spconv',
                norm_cfg=norm_cfg),
            _SparseBasicBlock(
                base_channel * 8,
                base_channel * 8,
                norm_cfg=norm_cfg,
                indice_key='res3'),
            _SparseBasicBlock(
                base_channel * 8,
                base_channel * 8,
                norm_cfg=norm_cfg,
                indice_key='res3'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel * 8, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True))

    def forward(self, voxel_features, coors, batch_size):
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features,
            coors.int(),
            self.sparse_shape_xyz[::-1],
            batch_size)
        x = self.conv_input(input_sp_tensor)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        dense = x.dense().permute(0, 1, 4, 3, 2).contiguous()
        return {'x': dense, 'pts_feats': [x]}
