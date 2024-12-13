import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import expit
from fds import FDS

print = logging.info


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, fds, bucket_num, bucket_start, start_update, start_smooth,
                 kernel, ks, sigma, momentum, ka=False, proj_dim=128, dropout=None, return_features=False):

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.ka = ka
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if not ka:
            self.linear = nn.Linear(512 * block.expansion, 1)
        else:
            self.linear = nn.Linear(512 * block.expansion, proj_dim)
        # self.proj = nn.Linear(512 * block.expansion, proj_dim)

        if fds:
            self.FDS = FDS(
                feature_dim=512 * block.expansion, bucket_num=bucket_num, bucket_start=bucket_start,
                start_update=start_update, start_smooth=start_smooth, kernel=kernel, ks=ks, sigma=sigma,
                momentum=momentum
            )
        self.fds = fds
        self.start_smooth = start_smooth

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, targets=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        encoding = x.view(x.size(0), -1)
        encoding_s = encoding

        if self.training and self.fds:
            if epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        if self.use_dropout:
            encoding_s = self.dropout(encoding_s)
        x = self.linear(encoding_s)

        if self.training and self.fds:
            return x, encoding
        else:
            return x, encoding


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


class KANet(nn.Module):
    def __init__(self, args, bucket_num=101, lr=0.001, readouts=1):
        super(KANet, self).__init__()
        self.encoder = resnet50(fds=False, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                start_update=args.start_update, start_smooth=args.start_smooth,
                                kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
                                ka=True, proj_dim=256)
        self.cls_head = torch.nn.Linear(2048, 1, bias=False)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        self.cls_head_opt = torch.optim.Adam(
            self.cls_head.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        # self.encoder_opt = torch.optim.Adam([
        #     {'params': self.encoder.parameters()},
        #     {'params': self.cls_head.parameters()}],
        #     lr=lr, betas=(0.5, 0.999), weight_decay=0.0001
        # )

        self.pairwise_certainty = PairwiseCertainty(bucket_num)

        self.loss = None
        self.item_1 = None
        self.item_2 = None

        self.items_n = bucket_num
        self.mse_loss = nn.MSELoss()
        # Create and initialise network layers
        # self.layer_1 = torch.nn.Linear(items_n, h1_size, bias=False)
        # self.layer_2 = torch.nn.Linear(h1_size, readouts, bias=False)

        self.register_parameter('W1', torch.nn.Parameter(torch.zeros(256, bucket_num)))
        self.register_parameter('W2', torch.nn.Parameter(torch.zeros(readouts, 256)))

        # torch.nn.init.normal_(self.layer_1.weight, std=w1_weight_std)
        # torch.nn.init.normal_(self.layer_2.weight, std=w2_weight_std)
        w1_weight_std = 0.025 * np.sqrt(1 / bucket_num)
        w2_weight_std = np.sqrt(1 / 256)
        torch.nn.init.normal_(self.W1, mean=0, std=w1_weight_std)
        torch.nn.init.normal_(self.W2, mean=0, std=w2_weight_std)

        self.non_linearity = nn.ReLU()
        pass

    def forward(self, x1, ):
        emb_x1, encoding = self.encoder(x1)
        # emb_x2, _ = self.encoder(x2)
        return emb_x1, encoding  # , emb_x2

        pass

    def update_w1(self, emb_x1, item_1, emb_x2, item_2, learning_rate, gamma):

        self.item_1 = item_1.long() - 1
        self.item_2 = item_2.long() - 1
        loss = 0
        for i in range(emb_x1.size()[0]):
            temp_1 = self.item_1[i] - 1
            temp_2 = self.item_2[i] - 1
            target = torch.tensor([1. if temp_1 > temp_2 else -1.]).cuda()

            # print(temp_2, temp_1)
            self.W1.data[:, temp_1] += 0.001 * emb_x1[i, :].unsqueeze(1)
            self.W1.data[:, temp_2] += 0.001 * emb_x2[i, :].unsqueeze(1)
            x1 = self._one_hot(temp_1)  # 还是原有的emb_x1
            x2 = self._one_hot(temp_2)  # 还是原有的emb_x2
            # h1 = self.non_linearity(torch.mm(self.W1, x1.unsqueeze(1)) - torch.mm(self.W1, x2.unsqueeze(1)))
            h1 = self.non_linearity(torch.mm(self.W1, x1.T) - torch.mm(self.W1, x2.T))
            out = torch.mm(self.W2, h1).squeeze(0)

            if self.W1.grad is not None:
                self.W1.grad = None
            if self.W2.grad is not None:
                self.W2.grad = None
            # print(f"{out.item(), target.item()}")
            loss = self.mse_loss(out, target)
            loss.backward()

            self.pairwise_certainty.update(temp_1, temp_2, loss.item(), gamma)

            items = [temp_1, temp_2]
            with torch.no_grad():
                for item, other_item in zip(items, items[::-1]):
                    # Calculate relative changes of weights
                    # import pdb
                    # pdb.set_trace()
                    w1 = self.W1
                    dw1 = -learning_rate * self.W1.grad[:, item]

                    # squeeze
                    w2 = self.W2[0]
                    dw2 = -learning_rate * self.W2.grad[0]

                    if torch.linalg.norm(dw2) > 1e-6 and torch.linalg.norm(dw1) > 1e-6:
                        # Apply corrections
                        dw1_ = dw1
                        nominator = (w2 @ dw1_ + dw2 @ w1[:, item] + dw2 @ dw1_ - dw2 @ w1)
                        denominator = (w2 @ dw1_ + dw2 @ dw1_)
                        cs = nominator / denominator
                        cs[item] = 0.
                        cs[other_item] = 0.
                        # certainty = torch.tensor(self.pairwise_certainty.a[:, item])
                        certainty = self.pairwise_certainty.a[:, item]
                        self.W1.data.add_(
                            -learning_rate * self.W1.grad + torch.outer(dw1_.squeeze(1), certainty.squeeze(1) * cs),
                            alpha=1)
                        # self.W1.data.add_(torch.outer(dw1_.squeeze(1), certainty.squeeze(1) * cs), alpha=1)
                        # self.W1.data.add_(
                        #     -learning_rate * self.W1.grad,
                        #     alpha=1)
                # w2 += dw2
            # self.W1 = w1
            # self.W1.data.add_(self.W1.grad, alpha=-learning_rate)
            self.W2.data.add_(self.W2.grad, alpha=-learning_rate)
        return loss.item()
        pass

    def update_encoder(self, embs, embs_gt):
        self.encoder_opt.zero_grad()
        loss = self.mse_loss(embs[0], embs_gt[0].detach()) + self.mse_loss(embs[1], embs_gt[1].detach())
        # loss =
        loss.backward()
        self.encoder_opt.step()
        return loss.item()

    def update_model(self, x1, item_1, x2, item_2, epoch=None):
        if epoch < 3:
            # print(f"x1.size(): {x1.size()}")
            emb_x1, encoding_x1 = self.forward(x1)
            emb_x2, encoding_x2 = self.forward(x2)
            # emb_x1, emb_x2 = self.forward(x1, item_1, x2, item_2)
            # target = torch.tensor([1. if item_1 > item_2 else -1.]).cuda()
            loss_ka = self.update_w1(emb_x1, item_1, emb_x2, item_2, learning_rate=0.01, gamma=0.5)

            embs_gt = [self.W1.data[:, item_1.squeeze().long() - 1].T, self.W1.data[:, item_1.squeeze().long() - 1].T]
            embs = [emb_x1, emb_x2]
            # import pdb
            # pdb.set_trace()
            # pred1 = self.cls_head(encoding_x1)
            # pred2 = self.cls_head(encoding_x2)
            loss_emb = self.update_encoder(embs, embs_gt, )
            return loss_emb, loss_ka
        else:
            self.cls_head_opt.zero_grad()
            _, encoding_x1 = self.forward(x1)
            _, encoding_x2 = self.forward(x2)
            pred1 = self.cls_head(encoding_x1)
            pred2 = self.cls_head(encoding_x2)
            loss = self.mse_loss(pred1, item_1) + self.mse_loss(pred2, item_2)
            loss.backward()
            self.cls_head_opt.step()
            return loss.item(),
        # KANet(self, x1, item_1, x2, item_2, epoch=None)

    def _one_hot(self, item):
        """ Create one-hot vector from index
        :param item: Item index
        :return: One-hot vector
        """
        x = torch.zeros(item.size(0), self.items_n).cuda()
        x[:, item.long()] = 1.
        return x


class PairwiseCertainty:
    def __init__(self, items_n, a=-1000., b=0.01):
        """Object to tract pairwise certainties

        :param items_n: Number of items
        :param a: Slope of sigmoidal
        :param b: Offset of sigmoidal
        """
        self.a = torch.from_numpy(np.zeros((items_n, items_n))).cuda()
        self.slope = a
        self.offset = b

    def update(self, i1, i2, loss, gamma):
        """Update certainty matrix based on performance of items a and b

        :param i1: Index of first item
        :param i2: Index of second item
        :param loss: Current loss value
        :param gamma: Time-constant of low-pass filter
        """
        certainty = PairwiseCertainty.phi(loss, self.slope, self.offset)
        aa = self.a
        aa_ = aa.clone()
        # import pdb
        # pdb.set_trace()

        self.a[i1, :] = (1. - gamma) * aa_[i1, :] + gamma * certainty * aa_[i2, :]
        self.a[:, i1] = (1. - gamma) * aa_[:, i1] + gamma * certainty * aa_[:, i2]

        self.a[:, i2] = (1. - gamma) * aa_[:, i2] + gamma * certainty * aa_[:, i1]
        self.a[i2, :] = (1. - gamma) * aa_[i2, :] + gamma * certainty * aa_[i1, :]

        self.a[i1, i2] = self.a[i2, i1] = (1. - gamma) * aa_[i1, i2] + gamma * certainty

    @staticmethod
    def phi(x, a, b):
        """Calculate certainty

        :param x: Current loss value
        :param a: Slope of sigmoidal
        :param b: Offset of sigmoidal
        :return: Certainty value
        """
        return expit(a * (x - b))


class contraNet_ada_emb(nn.Module):
    def __init__(self, args, proj_dim=128):
        super(contraNet_ada_emb, self).__init__()
        self.encoder = resnet50(fds=False, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                start_update=args.start_update, start_smooth=args.start_smooth,
                                kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
                                ka=False, proj_dim=proj_dim)
        # self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999),
        #                                     weight_decay=0.0001)
        self._avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_ctr = nn.Sequential(nn.Linear(512 * 4, proj_dim))
        self.cls_head = nn.Sequential(nn.Linear(proj_dim, 1), nn.ReLU())

    def forward(self, x1):
        # import pdb
        # pdb.set_trace()
        _, emb_x1 = self.encoder(x1)
        x_ctrst = self.fc_ctr(emb_x1)
        out = self.cls_head(x_ctrst)
        x_ctrst = F.normalize(x_ctrst, dim=1)
        return out, x_ctrst

    def fine_tune(self, x1):
        with torch.no_grad():
            _, emb_x1 = self.encoder(x1)
        x_ctrst = self.fc_ctr(emb_x1)
        y = self.cls_head(x_ctrst)
        return y


class contraNet_scl(nn.Module):
    def __init__(self, args, proj_dim=128):
        super(contraNet_scl, self).__init__()
        self.encoder = resnet50(fds=False, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                start_update=args.start_update, start_smooth=args.start_smooth,
                                kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
                                ka=False, proj_dim=proj_dim)
        # self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999),
        #                                     weight_decay=0.0001)
        self._avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_ctr = nn.Sequential(nn.Linear(512 * 4, 512 * 4), nn.ReLU(), nn.Linear(512 * 4, proj_dim))

    def forward(self, x1):
        out, emb_x1 = self.encoder(x1)
        x_ctrst = self.fc_ctr(emb_x1)
        x_ctrst = F.normalize(x_ctrst, dim=1)
        return out, x_ctrst

    def fine_tune(self, x1):
        with torch.no_grad():
            emb_x1 = self.encoder(x1)
        y = self.cls_head(emb_x1)
        return y


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    import pdb
    pdb.set_trace()
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    nn.parallel.gather(tensors_gather, "0")
    output = torch.cat(tensors_gather, dim=0)
    return output

# if __name__ == '__main__':
#     Network_my = Network_my(100, h1_size=128, w1_weight_std=0.5, w2_weight_std=0.5, non_linearity=torch.relu_, readouts=1)
