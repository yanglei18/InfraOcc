import torch


def bev_pool(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]
    out = feats.new_zeros((B, D, H, W, feats.shape[1]))
    out.index_put_(
        (coords[:, 3].long(), coords[:, 2].long(), coords[:, 0].long(), coords[:, 1].long()),
        feats,
        accumulate=True,
    )
    return out.permute(0, 4, 2, 3, 1).contiguous()
