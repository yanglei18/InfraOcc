import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)

kitti_class_names = [
    'empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign',
]


def inverse_sigmoid(x):
    x = x.to(torch.float32)
    if not torch.isfinite(x).all().item():
        raise FloatingPointError(
            'Non-finite probability reached semantic scaling loss')
    return torch.logit(x.clamp(1e-5, 1.0 - 1e-5))


def KL_sep(p, target):
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction='sum')
    return kl_term


def geo_scal_loss(pred, ssc_target, ignore_index=255, empty_idx=0, non_empty_idx=None):
    if non_empty_idx is not None:
        empty_idx = non_empty_idx

    pred = F.softmax(pred.float(), dim=1)
    empty_probs = pred[:, empty_idx]
    nonempty_probs = 1 - empty_probs

    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum() + eps)
    recall = intersection / (nonempty_target.sum() + eps)
    spec = ((1 - nonempty_target) * empty_probs).sum() / ((1 - nonempty_target).sum() + eps)
    with torch.amp.autocast('cuda', enabled=False):
        loss = (
            F.binary_cross_entropy_with_logits(inverse_sigmoid(precision), torch.ones_like(precision))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall), torch.ones_like(recall))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec), torch.ones_like(spec))
        )
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError(
                'Non-finite geometric scaling loss')
        return loss


def sem_scal_loss(pred_, ssc_target, ignore_index=255):
    with autocast('cuda', enabled=False):
        pred = F.softmax(pred_.float(), dim=1)
        loss = 0
        count = 0
        mask = ssc_target != ignore_index
        n_classes = pred.shape[1]
        begin = 1 if n_classes == 19 else 0
        for i in range(begin, n_classes - 1):
            p = pred[:, i]
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p) + 1e-5)
                    loss_class += F.binary_cross_entropy_with_logits(
                        inverse_sigmoid(precision), torch.ones_like(precision))
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target) + 1e-5)
                    loss_class += F.binary_cross_entropy_with_logits(
                        inverse_sigmoid(recall), torch.ones_like(recall))
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target) + 1e-5)
                    loss_class += F.binary_cross_entropy_with_logits(
                        inverse_sigmoid(specificity), torch.ones_like(specificity))
                loss += loss_class
        if count == 0:
            return pred.new_tensor(0.0)
        loss = loss / count
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError(
                'Non-finite semantic scaling loss')
        return loss


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    with autocast('cuda', enabled=False):
        criterion = nn.CrossEntropyLoss(
            weight=(class_weights.float()
                    if class_weights is not None else None),
            ignore_index=ignore_index,
            reduction='mean')
        loss = criterion(pred.float(), target.long())
    return loss


def vel_loss(pred, gt):
    with autocast('cuda', enabled=False):
        return F.l1_loss(pred, gt)
