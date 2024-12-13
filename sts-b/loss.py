import torch
import math
import torch.nn.functional as F
import numpy as np
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


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, inputs, targets, weights=None):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(inputs, targets, noise_var, weights)
        return loss


def bmc_loss(pred, target, noise_var, weight=None):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda(), weight=weight)
    loss = loss * (2 * noise_var).detach()

    return loss

class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")
        # sum_filt = torch.ones([1, 1, *win])
        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class SupConLoss_admargin(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_admargin, self).__init__()
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
        features = features.unsqueeze(1)
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
            mask = torch.eq(torch.round(labels, decimals=1), torch.round(labels, decimals=1).T).float().to(device)
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

        # import pdb
        # pdb.set_trace()
        # print(dist)
        # dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist.expand(-1, batch_size).to(device)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val), 0, 2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        ones_fullabdiff = torch.ones_like(dist_fullabdiff).to(device)

        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)

        mask = mask.repeat(anchor_count, contrast_count)

        adjn_abdiff = torch.multiply(torch.sub(ones_fullabdiff, mask), dist_fullabdiff)
        adj_abdiff = adjn_abdiff

        anchor_dot_contrast = (1 / self.temperature) * (anchor_dot_contrast + adj_abdiff)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0)
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

        # import pdb
        # pdb.set_trace()
        features = features.unsqueeze(1)
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
            mask = torch.eq(torch.round(labels, decimals=1), torch.round(labels, decimals=1).T).float().to(device)
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

        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0)
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


class SupConLoss_comp(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, max_inernal=100, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_comp, self).__init__()
        self.temperature = temperature
        self.max_inernal = max_inernal
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
        batch_size = labels.size()[0]
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # contrast_feature = F.normalize(contrast_feature, dim=1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # print(anchor_feature.size())


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
        sin_theta = torch.sqrt(1 - cos_theta ** 2 + 0.00001)
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
        # import pdb
        # pdb.set_trace()
        mask = mask * logits_mask

        # neg_logit = torch.div(cos_theta * cos_phi - sin_theta * sin_phi, temperature)
        neg_logit = torch.div(cos_theta * cos_phi - sin_theta * torch.abs(sin_phi), self.temperature)
        # neg_logit[mask == 1] = logits[mask == 1]
        exp_logits = torch.exp(neg_logit) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 0.00001) / (mask.sum(1) + 0.00001)
        # print(mean_log_prob_pos)
        # if torch.any(torch.isnan(mean_log_prob_pos)):
        #     import pdb
        #     pdb.set_trace()

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # import pdb
        # pdb.set_trace()
        return loss


class ACCon(torch.nn.Module):
    """Just for one batch!!!"""

    def __init__(self, max_inernal=5, temperature=1, contrast_mode='all',
                 base_temperature=1, tau=0.000001):
        super(ACCon, self).__init__()
        self.temperature = temperature
        self.max_inernal = max_inernal
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.tau = tau

    def forward(self, features, labels=None, mask=None, dist=None, norm_val=0.2, scale_s=150):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. 
        Returns:
            A loss scalar.
        """
        batch_size = features.size()[0]
        anchor_feature = features

        dist = (labels - labels.T).float().cuda()
        mask = torch.eq(torch.round(labels, decimals=1), torch.round(labels, decimals=1).T).float().cuda()
        phi = (1 - dist / self.max_inernal) * np.pi
        # print(phi)

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        cos_theta = torch.matmul(anchor_feature, anchor_feature.T)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        sin_theta = torch.sqrt(1 - cos_theta ** 2 + self.tau)

        logits = torch.div(cos_theta, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 1).view(-1, 1).cuda(),
            0
        )
        # print(logits_mask)

        neg_logit = torch.div(cos_theta * cos_phi - sin_theta * torch.abs(sin_phi), self.temperature)
        neg_logit[mask == 1] = logits[mask == 1]

        exp_logits = torch.exp(neg_logit)  * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # print(torch.sum(mask))
        mean_log_prob_pos = ((mask * log_prob).sum() + self.tau) / (mask.sum() + self.tau)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        if loss.item() < -100000 or loss.item() > 100000:
            raise "INF ERROR!!!"
            return
            import pdb
            pdb.set_trace()
        # import pdb
        # pdb.set_trace()
        # print(loss)
        return loss


# ConR loss function
def ConR(features, targets, preds, w=1, weights=1, t=0.2, e=0.01):
    t = 1
    # import pdb
    # pdb.set_trace()
    q = torch.nn.functional.normalize(features, dim=1)
    k = torch.nn.functional.normalize(features, dim=1)

    # import pdb
    # pdb.set_trace()
    targets = torch.round(targets, decimals=1).float().cuda()
    preds = torch.round(preds, decimals=1).float().cuda()
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
    neg = prod * neg_i

    pushing_w = weights * torch.exp(l_dist * e)
    neg_exp_dot = (pushing_w * (torch.exp(neg)) * neg_i).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom = pos_i.sum(1)

    loss = ((-torch.log(torch.div(torch.exp(pos), (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1))) * (pos_i)).sum(
        1) / denom)
    # print(loss)
    loss = (weights * (loss * no_neg_flag).unsqueeze(-1)).mean()

    return loss



def info_nce_loss(features):
    bsz, n_views, _ = features.size()
    device = ""
    labels = torch.cat([torch.arange(bsz) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / 0.05
    return logits, labels


if __name__ == '__main__':
    import torch.nn as nn
    import torch
    features = torch.rand((16,  24)) #.cuda()
    labels = torch.randint(0, 3, (16, 1)) # .cuda()
    print(labels)
    lossss = SupConLoss_comp_v3()
    v = lossss(features, labels)
    print(v)