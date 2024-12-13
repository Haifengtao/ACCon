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


class contraNet(nn.Module):
    def __init__(self, args, bucket_num=100, lr=0.001, proj_dim=256, readouts=1):
        super(contraNet, self).__init__()
        self.encoder = resnet50(fds=False, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                start_update=args.start_update, start_smooth=args.start_smooth,
                                kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
                                ka=True, proj_dim=proj_dim)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999),
                                            weight_decay=0.0001)

        self.pairwise_certainty = PairwiseCertainty(bucket_num)

        self.loss = None
        self.item_1 = None
        self.item_2 = None

        # Create and initialise network layers
        self.cls_head = torch.nn.Linear(proj_dim, readouts, bias=False)

        self.register_parameter('W1', torch.nn.Parameter(torch.zeros(256, bucket_num)))
        self.register_parameter('W2', torch.nn.Parameter(torch.zeros(readouts, 256)))

        # torch.nn.init.normal_(self.layer_1.weight, std=w1_weight_std)
        # torch.nn.init.normal_(self.layer_2.weight, std=w2_weight_std)
        w1_weight_std = 0.025 * np.sqrt(1 / bucket_num)
        w2_weight_std = np.sqrt(1 / 256)
        torch.nn.init.normal_(self.W1, mean=0, std=w1_weight_std)
        torch.nn.init.normal_(self.W2, mean=0, std=w2_weight_std)

    def forward(self, x1, item_1, x2, item_2):
        _, emb_x1 = self.encoder(x1)
        _, emb_x2 = self.encoder(x2)

        h1 = self.non_linearity(emb_x1 - emb_x2)

        out = torch.mm(self.W2, h1).squeeze(0)
        # y = self.ka_net(emb_x1-emb_x2)
        return h1, out, item_2 - item_1

    def fine_tune(self, x1):
        with torch.no_grad():
            emb_x1 = self.encoder(x1)
        y = self.cls_head(emb_x1)
        return y


class contransNet_NCC(nn.Module):
    """
    SCL menthod
    """

    def __init__(self, args, bucket_num=100, lr=0.001, proj_dim=256, readouts=1):
        super(contransNet_NCC, self).__init__()
        self.encoder = resnet50(fds=False, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                start_update=args.start_update, start_smooth=args.start_smooth,
                                kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
                                ka=True, proj_dim=proj_dim)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999),
                                            weight_decay=0.0001)

        self.pairwise_certainty = PairwiseCertainty(bucket_num)
        self.proj_dim = proj_dim
        self.loss = None
        self.item_1 = None
        self.item_2 = None

        # Create and initialise network layers
        self.cls_head = torch.nn.Linear(512 * 4, readouts, bias=False)
        self.proj = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 4, proj_dim, bias=False))
        self.read_out = torch.nn.Linear(proj_dim, 1, bias=False)

        self.register_parameter('W1', torch.nn.Parameter(torch.zeros(256, bucket_num)))
        self.register_parameter('W2', torch.nn.Parameter(torch.zeros(readouts, 256)))

        # torch.nn.init.normal_(self.layer_1.weight, std=w1_weight_std)
        # torch.nn.init.normal_(self.layer_2.weight, std=w2_weight_std)
        w1_weight_std = 0.025 * np.sqrt(1 / bucket_num)
        w2_weight_std = np.sqrt(1 / 256)
        torch.nn.init.normal_(self.W1, mean=0, std=w1_weight_std)
        torch.nn.init.normal_(self.W2, mean=0, std=w2_weight_std)

    def forward(self, x1, item_1=None):
        # import pdb
        # pdb.set_trace()
        size_ = x1.size()
        device = x1.device
        _, emb_x1 = self.encoder(x1)
        proj_emb = self.proj(emb_x1)
        proj_emb = nn.functional.normalize(proj_emb, dim=1)  # [256, 256]
        y = self.cls_head(emb_x1)
        if item_1 is None:
            return y, 0
        else:
            # import pdb
            # pdb.set_trace()
            # proj_emb = proj_emb.T.unsqueeze(2)
            # temp_ones = torch.ones((self.proj_dim, size_[0], 1)).to(device)
            # sub_ = torch.bmm(proj_emb, torch.permute(temp_ones, (0, 2, 1))) - torch.bmm(temp_ones, torch.permute(proj_emb, (0, 2, 1)))
            # simi_matrix = self.read_out(torch.permute(sub_, (2, 1, 0))).squeeze(2)
            # temp_ones = torch.ones((size_[0], 1)).to(device)
            # target_matrix = torch.mm(item_1, temp_ones.T) - torch.mm(temp_ones, item_1.T)
            simi_matrix = torch.mm(proj_emb, proj_emb.T)
            temp_ones = torch.ones((size_[0], 1)).to(device)
            target_matrix = 1 / (torch.abs(torch.mm(item_1, temp_ones.T) - torch.mm(temp_ones, item_1.T)) + 1)
            return y, [simi_matrix, target_matrix]

            # return y, [simi_matrix, target_matrix]

    def fine_tune(self, x1):
        with torch.no_grad():
            emb_x1 = self.encoder(x1)
        y = self.cls_head(emb_x1)
        return y


class contransNet_NCC_v1(nn.Module):
    """
    SCL menthod
    """

    def __init__(self, args, bucket_num=100, lr=0.001, proj_dim=256, readouts=1):
        super(contransNet_NCC_v1, self).__init__()
        self.encoder = resnet50(fds=False, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                start_update=args.start_update, start_smooth=args.start_smooth,
                                kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
                                ka=True, proj_dim=proj_dim)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999),
                                            weight_decay=0.0001)

        self.pairwise_certainty = PairwiseCertainty(bucket_num)
        self.proj_dim = proj_dim
        self.loss = None
        self.item_1 = None
        self.item_2 = None

        # Create and initialise network layers
        self.cls_head = torch.nn.Linear(512 * 4, readouts, bias=False)
        self.proj = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 4, proj_dim, bias=False))
        self.read_out = torch.nn.Linear(proj_dim, 1, bias=False)

        self.register_parameter('W1', torch.nn.Parameter(torch.zeros(256, bucket_num)))
        self.register_parameter('W2', torch.nn.Parameter(torch.zeros(readouts, 256)))

        # torch.nn.init.normal_(self.layer_1.weight, std=w1_weight_std)
        # torch.nn.init.normal_(self.layer_2.weight, std=w2_weight_std)
        w1_weight_std = 0.025 * np.sqrt(1 / bucket_num)
        w2_weight_std = np.sqrt(1 / 256)
        torch.nn.init.normal_(self.W1, mean=0, std=w1_weight_std)
        torch.nn.init.normal_(self.W2, mean=0, std=w2_weight_std)

    def forward(self, x1, item_1=None):
        # import pdb
        # pdb.set_trace()
        size_ = x1.size()
        device = x1.device
        _, emb_x1 = self.encoder(x1)
        proj_emb = self.proj(emb_x1)
        proj_emb = nn.functional.normalize(proj_emb, dim=1)  # [256, 256]
        y = self.cls_head(emb_x1)
        if item_1 is None:
            return y, 0
        else:
            # import pdb
            # pdb.set_trace()
            # proj_emb = proj_emb.T.unsqueeze(2)
            # temp_ones = torch.ones((self.proj_dim, size_[0], 1)).to(device)
            # sub_ = torch.bmm(proj_emb, torch.permute(temp_ones, (0, 2, 1))) - torch.bmm(temp_ones, torch.permute(proj_emb, (0, 2, 1)))
            # simi_matrix = self.read_out(torch.permute(sub_, (2, 1, 0))).squeeze(2)
            # temp_ones = torch.ones((size_[0], 1)).to(device)
            # target_matrix = torch.mm(item_1, temp_ones.T) - torch.mm(temp_ones, item_1.T)
            simi_matrix = torch.mm(proj_emb, proj_emb.T) + 1
            temp_ones = torch.ones((size_[0], 1)).to(device)
            target_matrix = 1 / (torch.abs(torch.mm(item_1, temp_ones.T) - torch.mm(temp_ones, item_1.T)) + 1)
            # target_matrix = (torch.mm(item_1, temp_ones.T) - torch.mm(temp_ones, item_1.T) + 100)/100
            return y, [simi_matrix, target_matrix]

            # return y, [simi_matrix, target_matrix]

    def fine_tune(self, x1):
        with torch.no_grad():
            emb_x1 = self.encoder(x1)
        y = self.cls_head(emb_x1)
        return y


class contransNet_v2(nn.Module):
    def __init__(self, args, bucket_num=100, lr=0.001, proj_dim=256, readouts=1):
        super(contransNet_v2, self).__init__()
        self.encoder = resnet50(fds=False, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                start_update=args.start_update, start_smooth=args.start_smooth,
                                kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
                                ka=True, proj_dim=proj_dim)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999),
                                            weight_decay=0.0001)

        self.pairwise_certainty = PairwiseCertainty(bucket_num)
        self.proj_dim = proj_dim
        self.loss = None
        self.item_1 = None
        self.item_2 = None

        # Create and initialise network layers
        self.cls_head = torch.nn.Linear(512 * 4, readouts, bias=False)
        self.proj = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 4, proj_dim, bias=False))
        self.read_out = torch.nn.Linear(proj_dim, 1, bias=False)

        self.register_parameter('W1', torch.nn.Parameter(torch.zeros(256, bucket_num)))
        self.register_parameter('W2', torch.nn.Parameter(torch.zeros(readouts, 256)))

        # torch.nn.init.normal_(self.layer_1.weight, std=w1_weight_std)
        # torch.nn.init.normal_(self.layer_2.weight, std=w2_weight_std)
        w1_weight_std = 0.025 * np.sqrt(1 / bucket_num)
        w2_weight_std = np.sqrt(1 / 256)
        torch.nn.init.normal_(self.W1, mean=0, std=w1_weight_std)
        torch.nn.init.normal_(self.W2, mean=0, std=w2_weight_std)

    def forward(self, x1, item_1=None):
        # import pdb
        # pdb.set_trace()
        size_ = x1.size()
        device = x1.device
        _, emb_x1 = self.encoder(x1)

        proj_emb = self.proj(emb_x1)
        proj_emb = nn.functional.normalize(proj_emb, dim=1)  # [256, 256]
        y = self.cls_head(emb_x1)
        if item_1 is None:
            return y, 0
        else:
            # import pdb
            # pdb.set_trace()
            proj_emb = proj_emb.T.unsqueeze(2)
            temp_ones = torch.ones((self.proj_dim, size_[0], 1)).to(device)
            sub_ = torch.bmm(proj_emb, torch.permute(temp_ones, (0, 2, 1))) - torch.bmm(temp_ones,
                                                                                        torch.permute(proj_emb,
                                                                                                      (0, 2, 1)))
            simi_matrix = self.read_out(torch.permute(sub_, (2, 1, 0))).squeeze(2)
            temp_ones = torch.ones((size_[0], 1)).to(device)
            target_matrix = torch.mm(item_1, temp_ones.T) - torch.mm(temp_ones, item_1.T)
            # simi_matrix = torch.mm(proj_emb, proj_emb.T)
            # temp_ones = torch.ones((size_[0], 1)).to(device)
            # target_matrix = 1/(torch.abs(torch.mm(item_1, temp_ones.T) - torch.mm(temp_ones, item_1.T)) + 1)
            # return y, [simi_matrix, target_matrix]
            return y, [simi_matrix, target_matrix]

    def fine_tune(self, x1):
        with torch.no_grad():
            emb_x1 = self.encoder(x1)
        y = self.cls_head(emb_x1)
        return y


class Network_my(torch.nn.Module):
    def __init__(self, items_n, h1_size, w1_weight_std, w2_weight_std, non_linearity=torch.relu_, readouts=1):
        """Two-layer neural network to calculate the pairwise relationships of two items. In particular, the network
        function is W_2 sigma(W_1x_1 - W_1x_2), where x_1 and x_2 are one-hot vectors that indicate the index of the
        presented item. The network either has one or two readout heads to either jointly or independently encode the
        relationship between items.

        :param items_n: Number of items
        :param h1_size: Size of hidden layer
        :param w1_weight_std: Standard deviation of initial weights of input layer
        :param w2_weight_std: Standard deviation of initial weights of readout layer
        :param non_linearity: Non-linearity to be applied on the hidden layer representation
        :param readouts: Number of readout heads (1 or 2)
        """
        super().__init__()
        self.items_n = items_n
        self.h1_size = h1_size
        self.non_linearity = non_linearity
        self.readouts = readouts

        self.pairwise_certainty = PairwiseCertainty(items_n)

        self.loss = nn.MSELoss()
        self.item_1 = None
        self.item_2 = None

        # Create and initialise network layers
        # self.layer_1 = torch.nn.Linear(items_n, h1_size, bias=False)
        # self.layer_2 = torch.nn.Linear(h1_size, readouts, bias=False)

        self.register_parameter('W1', torch.nn.Parameter(torch.zeros(h1_size, items_n)))
        self.register_parameter('W2', torch.nn.Parameter(torch.zeros(readouts, h1_size)))

        # torch.nn.init.normal_(self.layer_1.weight, std=w1_weight_std)
        # torch.nn.init.normal_(self.layer_2.weight, std=w2_weight_std)
        torch.nn.init.normal_(self.W1, mean=0, std=w1_weight_std)
        torch.nn.init.normal_(self.W2, mean=0, std=w2_weight_std)

    def forward(self, item_1, item_2):
        """Calculate the network output

        :param item_1: Index of the first item
        :param item_2: Index of the second item
        :return:
        """
        self.item_1 = item_1
        self.item_2 = item_2

        x1 = self._one_hot(item_1)
        x2 = self._one_hot(item_2)

        self.correct(0.01, gamma=0.5)
        h1 = self.non_linearity(torch.mm(self.W1, x1.unsqueeze(1)) - torch.mm(self.W1, x2.unsqueeze(1)))
        out = torch.mm(self.W2, h1).squeeze(0)
        return h1, out

    def correct(self, learning_rate, gamma):
        """Preserve previously learned relationships between items dependent on the certainty that two items are
        correctly related in embedding space.

        :param learning_rate: Learning rate of gradient descent
        :param gamma: Time-constant of low-pass filter to acquire certainties
        :return:
        """
        self.pairwise_certainty.update(self.item_1, self.item_2, self.loss().item(), gamma)
        items = [self.item_1, self.item_2]

        with torch.no_grad():
            for item, other_item in zip(items, items[::-1]):
                # Calculate relative changes of weights
                w1 = self.W1
                dw1 = -learning_rate * self.W1.grad[:, item]
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
                    certainty = torch.tensor(self.pairwise_certainty.a[:, item])
                    w1 += torch.outer(dw1_, certainty * cs)
            # w2 += dw2
        self.W1 = w1
        self.W1.data.add_(self.W1.grad, alpha=-learning_rate)
        self.W2.data.add_(self.W2.grad, alpha=-learning_rate)

    def evaluate(self):
        """Evaluate network output on all possible combinations of input items

        :return: Numpy array of size (items_n, items_n) containing all network outputs
        """
        with torch.no_grad():
            n = self.items_n
            grid = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    with torch.no_grad():
                        if self.readouts == 1:
                            grid[j, i] = self.forward(i, j)[1].item()
                        elif self.readouts == 2:
                            grid[j, i] = self.forward(i, j)[1][0].item() - self.forward(i, j)[1][1].item()
            return grid

    def extract_h1s(self):
        """Calculate network hidden state representations for all input items

        :return: Numpy array of size (items_n, h1_size) containing all hidden states
        """
        with torch.no_grad():
            n = self.items_n
            h1s = np.zeros((n, self.h1_size))
            for i in range(n):
                with torch.no_grad():
                    h1s[i, :] = self.layer_1.weight[:, i].detach().numpy().copy()
            return h1s

    def _one_hot(self, item):
        """ Create one-hot vector from index

        :param item: Item index
        :return: One-hot vector
        """
        x = torch.zeros(self.items_n)
        x[item] = 1.
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


class contraNet_ada(nn.Module):
    def __init__(self, args, proj_dim=128):
        super(contraNet_ada, self).__init__()
        self.encoder = resnet50(fds=False, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                start_update=args.start_update, start_smooth=args.start_smooth,
                                kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
                                ka=False, proj_dim=proj_dim)
        # self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999),
        #                                     weight_decay=0.0001)
        self._avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_ctr = nn.Sequential(nn.Linear(512 * 4, 512 * 4), nn.ReLU(), nn.Linear(512 * 4, proj_dim))
        self.cls_head = nn.Sequential(nn.Linear(512 * 4, 1), nn.ReLU())

    def forward(self, x1):
        out, emb_x1 = self.encoder(x1)
        x_ctrst = self.fc_ctr(emb_x1)
        x_ctrst = F.normalize(x_ctrst, dim=1)
        return nn.functional.relu(out), x_ctrst

    def fine_tune(self, x1):
        with torch.no_grad():
            _, emb_x1 = self.encoder(x1)
        y = self.cls_head(emb_x1)
        return y


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


class contraNet_ada_queue(nn.Module):
    def __init__(self, args, proj_dim=128):
        super(contraNet_ada_queue, self).__init__()
        self.encoder = resnet50(fds=False, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                start_update=args.start_update, start_smooth=args.start_smooth,
                                kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
                                ka=False, proj_dim=proj_dim)
        # self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999),
        #                                     weight_decay=0.0001)
        self._avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_ctr = nn.Sequential(nn.Linear(512 * 4, 512 * 4), nn.ReLU(), nn.Linear(512 * 4, proj_dim))
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
        y = self.cls_head(emb_x1)
        return y


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, args, dim=128, K=65536, m=0.999, temperature=0.05, max_inernal=100):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.temperature = temperature
        self.base_temperature = temperature
        self.max_inernal = max_inernal
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                  start_update=args.start_update, start_smooth=args.start_smooth,
                                  kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)
        self.encoder_k = resnet50(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                                  start_update=args.start_update, start_smooth=args.start_smooth,
                                  kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)
        self.fc_ctr = nn.Sequential(nn.Linear(512 * 4, 512 * 4), nn.ReLU(), nn.Linear(512 * 4, dim))
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_label", torch.zeros(1, K))
        self.register_buffer("queue_label_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def _deLqueue_and_enLqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_label_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_label[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_label_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def inter(self, x1):
        out, emb_x1 = self.encoder_q(x1)
        return nn.functional.relu(out)

    def forward(self, x1, y1):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # import pdb
        # pdb.set_trace()
        bs = x1.size()[0]
        im_q, im_k = x1[:bs // 2, ...], x1[bs // 2:, ...]
        y_q = y1[:bs//2, ...]
        # print(y1.size(), im_q.size())

        # compute query features
        # import pdb
        # pdb.set_trace()
        out, emb_x1 = self.encoder_q(im_q)
        x_ctrst = self.fc_ctr(emb_x1)
        q = F.normalize(x_ctrst, dim=1)
        # return nn.functional.relu(out), x_ctrst

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            _, emb_k1 = self.encoder_k(im_k)
            k_ctrst = self.fc_ctr(emb_k1)
            k = F.normalize(k_ctrst, dim=1)
            # k = self.encoder_k(im_k)  # keys: NxC
            # k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # import pdb
        # pdb.set_trace()
        contrast_feature = torch.cat([k.T, self.queue.clone().detach()], dim=1)
        label_queue = torch.cat([y_q.T, self.queue_label.clone().detach()], dim=1)
        # contrast_feature = F.normalize(contrast_feature, dim=1)
        # print(contrast_feature.size())
        anchor_feature = q
        # print(anchor_feature.size())

        dist = (y_q - label_queue).float().cuda()
        # print(labels)
        # print(dist)
        mask = torch.eq(dist, 0).float().cuda()
        # print(dist.size())
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

        cos_theta = torch.matmul(anchor_feature, contrast_feature)

        # print(cos_theta.size(), pos_neg.size())
        # print(1 - cos_theta ** 2)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        # sin_theta = pos_neg * torch.sqrt(1 - cos_theta ** 2)
        sin_theta = torch.sqrt(1 - cos_theta ** 2 + 0.00000001)
        # print("sin_theta", sin_theta)

        logits = torch.div(cos_theta, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # print(mask.size())
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(bs//2).view(-1, 1).cuda(),
        #     0
        # )
        # import pdb
        # pdb.set_trace()
        # mask2 = mask * logits_mask

        # neg_logit = torch.div(cos_theta * cos_phi - sin_theta * sin_phi, temperature)
        # import pdb
        # pdb.set_trace()
        # neg_logit = torch.div(cos_theta * cos_phi - sin_theta * torch.abs(sin_phi), self.temperature)
        # neg_logit[mask == 1] = logits[mask == 1]
        # exp_logits = torch.exp(neg_logit) * logits_mask
        #
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mean_log_prob_pos = ((mask * log_prob).sum(1) + 0.00000001) / (mask.sum(1) + 0.00000001)

        neg_logit = torch.div(cos_theta * cos_phi - sin_theta * torch.abs(sin_phi), self.temperature)
        neg_logit[mask == 1] = logits[mask == 1]
        exp_logits = torch.exp(neg_logit)  # * logits_mask

        # import pdb
        # pdb.set_trace()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 0.00000001) / (mask.sum(1) + 0.00000001)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        con_loss = loss.view(bs//2).mean()

        # # compute logits
        # # Einstein sum is more intuitive
        # # positive logits: Nx1
        # l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # # negative logits: NxK
        # l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        #
        # # logits: Nx(1+K)
        # logits = torch.cat([l_pos, l_neg], dim=1)
        #
        # # apply temperature
        # logits /= self.T
        #
        # # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._deLqueue_and_enLqueue(y_q)
        # print(y_q, self.queue_label)
        return nn.functional.relu(out), con_loss


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
