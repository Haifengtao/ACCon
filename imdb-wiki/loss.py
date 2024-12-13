import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, dist=None, norm_val=0.2, scale_s=150):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # contrast_feature = F.normalize(contrast_feature, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            # print(anchor_count)
            # exit()
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        mask = mask.repeat(anchor_count, contrast_count)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # import pdb
        # pdb.set_trace()
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 0.00001) / (mask.sum(1) + 0.00001)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print()
        return loss


class ACCon(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, max_inernal=100, temperature=1, contrast_mode='all',
                 base_temperature=1, tau=0.000001):
        super(ACCon, self).__init__()
        self.temperature = temperature
        self.max_inernal = max_inernal
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.tau = tau
    def forward(self, features, labels=None, mask=None, dist=None, norm_val=0.2, scale_s=150):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # import pdb
        # pdb.set_trace()
        batch_size = features.size()[0]
        # if features.size()[0] != labels.size()[0]:
        #     fs = torch.split(labels, [batch_size, ] * features.size()[1], dim=0)
        #     labels = torch.cat([f1.unsqueeze(1) for f1 in fs], dim=1)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        dist = (labels - labels.T).float().cuda()
        # print(labels)
        # print(dist)
        mask = torch.eq(labels, labels.T).float().cuda()
        dist = dist.repeat(anchor_count, contrast_count)
        mask = mask.repeat(anchor_count, contrast_count)

        phi = (1 - dist / self.max_inernal) * np.pi
        # print(phi)

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        # print(cos_phi)
        # print(sin_phi)

        # pos_neg = torch.zeros_like(dist)
        # pos_neg[dist > 0] = 1
        # pos_neg[dist < 0] = -1
        # print("a", pos_neg)

        cos_theta = torch.matmul(anchor_feature, contrast_feature.T)

        # print(cos_theta.size(), pos_neg.size())
        # print(1 - cos_theta ** 2)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        # sin_theta = pos_neg * torch.sqrt(1 - cos_theta ** 2)
        sin_theta = torch.sqrt(1 - cos_theta ** 2 + self.tau)
        # print("sin_theta", sin_theta)

        logits = torch.div(cos_theta, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )

        mask = mask * logits_mask

        neg_logit = torch.div(cos_theta * cos_phi - sin_theta * torch.abs(sin_phi), self.temperature)
        neg_logit[mask == 1] = logits[mask == 1]
        exp_logits = torch.exp(neg_logit) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = ((mask * log_prob).sum(1) + self.tau) / (mask.sum(1) + self.tau)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

# ConR loss function
def ConR(features, targets, preds, w=1, weights=1, t=0.2, e=0.01):
    t = 0.07

    q = torch.nn.functional.normalize(features, dim=1)
    k = torch.nn.functional.normalize(features, dim=1)

    l_k = targets.flatten()[None, :]
    l_q = targets

    p_k = preds.flatten()[None, :]
    p_q = preds

    # label distance as a coefficient for neg samples
    eta = e * weights

    l_dist = torch.abs(l_q - l_k)
    p_dist = torch.abs(p_q - p_k)

    pos_i = l_dist.le(w)
    neg_i = ((~ (l_dist.le(w))) * (p_dist.le(w)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0

    prod = torch.einsum("nc,kc->nk", [q, k]) / t
    pos = prod * pos_i
    neg = prod * neg_i

    pushing_w = weights * torch.exp(l_dist * e)
    neg_exp_dot = (pushing_w * (torch.exp(neg)) * neg_i).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom = pos_i.sum(1)

    loss = ((-torch.log(torch.div(torch.exp(pos), (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1))) * (pos_i)).sum(
        1) / denom)

    loss = (weights * (loss * no_neg_flag).unsqueeze(-1)).mean()

    return loss


def acConR(features, targets, preds, w=1, weights=1, t=0.2, e=0.01):
    t = 0.07
    epsilon = 0.000001
    q = torch.nn.functional.normalize(features, dim=1)
    k = torch.nn.functional.normalize(features, dim=1)

    l_k = targets.flatten()[None, :]
    l_q = targets

    p_k = preds.flatten()[None, :]
    p_q = preds

    # label distance as a coefficient for neg samples
    eta = e * weights

    l_dist = torch.abs(l_q - l_k)
    p_dist = torch.abs(p_q - p_k)

    pos_i = l_dist.le(w)     # 真实的距离小于1的
    neg_i = ((~ (l_dist.le(w))) * (p_dist.le(w)))  # 真实的距离大于w，并且估计的是错误的

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0

    prod = torch.einsum("nc,kc->nk", [q, k]) / t
    pos = prod * pos_i
    # neg = prod * neg_i

    # acCON
    max_inernal = 100
    phi = (1 - l_dist / max_inernal) * np.pi
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = prod
    cos_theta = torch.clamp(cos_theta, -1, 1)
    sin_theta = torch.sqrt(1 - cos_theta ** 2 + epsilon)
    neg_logit = (cos_theta * cos_phi - sin_theta * torch.abs(sin_phi))/t
    neg_exp_dot = ((torch.exp(neg_logit * neg_i)) * neg_i).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom = pos_i.sum(1)
    loss = ((-torch.log(torch.div(torch.exp(pos), (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1))) * (pos_i)).sum(
             1) / denom)
    # print(loss)
    loss = (weights * (loss * no_neg_flag).unsqueeze(-1)).mean()

    return loss

