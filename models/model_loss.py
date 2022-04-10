import torch
import torch.nn.functional as F
from models import CenterNetGT

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index  = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def reg_l1_loss(output, mask, index, target):
    pred = gather_feature(output, index, use_transform=True)
    mask = mask.unsqueeze(dim=2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def modified_focal_loss(pred, gt):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def centernetloss(pred_dict, gt_dict):
    # scoremap loss
    pred_score = pred_dict['cls']
    cur_device = pred_score.device
    for k in gt_dict:
        gt_dict[k] = gt_dict[k].to(cur_device)

    loss_cls = modified_focal_loss(pred_score, gt_dict['score_map'])

    mask = gt_dict['reg_mask']
    index = gt_dict['index']
    index = index.to(torch.long)

    # regression loss
    loss_reg = reg_l1_loss(pred_dict['reg'], mask, index, gt_dict['reg'])

    # loss_cls *= 1
    # loss_reg *= 1

    loss = {
        "loss_cls": loss_cls,
        "loss_center_reg": loss_reg,
    }
    return loss


@torch.no_grad()
def get_ground_truth(batched_inputs):
    return CenterNetGT.generate(batched_inputs)