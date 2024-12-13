import logging
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
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
                 kernel, ks, sigma, momentum, dropout=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512 * block.expansion, 1)

        if fds:
            self.FDS = FDS(
                feature_dim=512 * block.expansion, bucket_num=bucket_num, bucket_start=bucket_start,
                start_update=start_update, start_smooth=start_smooth, kernel=kernel, ks=ks, sigma=sigma, momentum=momentum
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

        # if self.training and self.fds:
        #     return x, encoding
        # else:
        #     return x
        return x, encoding

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


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
                                )
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

