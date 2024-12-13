# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
########################################################################################
# Code is based on the LDS and FDS (https://arxiv.org/pdf/2102.09554.pdf) implementation
# from https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir
# by Yuzhe Yang et al.
########################################################################################
import time
import argparse
import logging
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
from scipy.stats import gmean
from collections import defaultdict
import datetime
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from loss import *

# from balanaced_mse import *
from utils import *
from datasets import AgeDB, AgeDB_AdaSCL
from resnet import resnet50, contraNet_ada, contraNet_ada_emb, MoCo
# from ranksim import batchwise_ranking_regularizer
import os


os.environ["KMP_WARNINGS"] = "FALSE"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# imbalanced related
# LDS
parser.add_argument('--random_seed', default=42, help='whether to enable LDS')
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
# FDS
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=9, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
parser.add_argument('--bucket_start', type=int, default=3, choices=[0, 3],
                    help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'],
                    help='cost-sensitive reweighting scheme')
# two-stage training: RRT
parser.add_argument('--retrain_fc', action='store_true', default=False,
                    help='whether to retrain last regression layer (regressor)')

# mine
parser.add_argument('--proj_dims', type=int, default=128, help='regularization_type')
parser.add_argument('--temperature', type=float, default=0.05, help='regularization_type')
# parser.add_argument('--regularization_weight', type=float, default=1, help='weight of the regularization term')


# batchwise ranking regularizer
parser.add_argument('--regularization_type', default='comp2', choices=['scl', 'ada', 'comp', "rr_acSCL", 'RNC', 'conR', 'comp2', 'rank'], help='regularization_type')
parser.add_argument('--regularization_weight', type=float, default=1, help='weight of the regularization term')
parser.add_argument('--interpolation_lambda', type=float, default=2, help='interpolation strength')
parser.add_argument('--K', type=int, default=128, help='queue size')
parser.add_argument('--dims', type=int, default=128, help='queue size')


# training/optimization related
parser.add_argument('--dataset', type=str, default='agedb', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--datatype', type=str, default='natural',
                    help='the type of test dataset') # choices=['natural', 'balanced']
parser.add_argument('--n_views', type=int, default=2, help='number of views for contrastive learning')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
parser.add_argument('--model', type=str, default='contraNet_ada', help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['bmse', 'mse', 'l1', 'focal_l1', 'focal_mse', 'huber'],
                    help='training loss type')
parser.add_argument('--init_noise_sigma', type=float, default=1., help='initial scale of the noise')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=1, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 120], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--print_freq', type=int, default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--workers', type=int, default=0, help='number of workers used in data loading')

# checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--gpu', type=str, default='0', help='checkpoint file path to resume training')
parser.add_argument('--pretrained', type=str, default='', help='checkpoint file path to load backbone weights')
parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()
args.start_epoch, args.best_loss = 0, 1e5

seed = random.randint(1, 10000) if args.random_seed < 0 else args.random_seed
random.seed(seed)
torch.manual_seed(seed)


if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
if args.retrain_fc:
    args.store_name += f'_retrain_fc'
if args.regularization_weight > 0:
    args.store_name += f'_reg_{args.regularization_type}_wight_{args.regularization_weight}'
args.store_name = f"{args.datatype}_{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"

timestamp = str(datetime.datetime.now())
timestamp = '-'.join(timestamp.split(' '))
args.store_name = args.store_name  # + '_' + timestamp

prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")


def main():
    # if args.gpu is not None:
    #     print(f"Use GPU: {args.gpu} for training")

    # Data
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}_{args.datatype}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}_{args.datatype}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']

    train_dataset = AgeDB_AdaSCL(data_dir=args.data_dir, n_views=args.n_views, df=df_train, img_size=args.img_size,
                                 split='train',
                                 reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks,
                                 lds_sigma=args.lds_sigma)
    val_dataset = AgeDB_AdaSCL(data_dir=args.data_dir, n_views=args.n_views, df=df_val, img_size=args.img_size,
                               split='val')
    test_dataset = AgeDB_AdaSCL(data_dir=args.data_dir, n_views=args.n_views, df=df_test, img_size=args.img_size,
                                split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model
    print('=====> Building model...')
    if args.model == "resnet50":
        model = resnet50(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                         start_update=args.start_update, start_smooth=args.start_smooth,
                         kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)
        model = torch.nn.DataParallel(model).cuda()

    if args.model == "resnet18":
        model = resnet50(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                         start_update=args.start_update, start_smooth=args.start_smooth,
                         kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)
        model = torch.nn.DataParallel(model).cuda()
    # elif args.model == "contraNet_ada":
    #     model = contraNet_ada(args, proj_dim=128)
    #     model = torch.nn.DataParallel(model).cuda()
    elif args.model == "contraNet_ada_emb":
        model = contraNet_ada_emb(args, proj_dim=args.proj_dims)
        model = torch.nn.DataParallel(model).cuda()

    elif args.model == "contraNet_ada":
        model = contraNet_ada(args, proj_dim=args.proj_dims)
        model = torch.nn.DataParallel(model).cuda()

    # elif args.model == "contraNet_ada_queue":
    #     model = MoCo(args, dim=args.dims, K=args.K, m=0.999, temperature=0.05)
    #
    #     model = torch.nn.DataParallel(model).cuda()
        # model = model.cuda()

        # 1) 初始化

    # if args.model == "contraNet_ada_queue":
    #     Con_Loss = nn.Identity(dist=None, norm_val=None, scale_s=20)
    if args.regularization_type == "ada":
        Con_Loss = SupConLoss_admargin(temperature=args.temperature, base_temperature=args.temperature)
    elif args.regularization_type == "scl":
        Con_Loss = SupConLoss(temperature=args.temperature, base_temperature=args.temperature)
    # elif args.regularization_type == "comp":
    #     Con_Loss = SupConLoss_comp(temperature=args.temperature, base_temperature=args.temperature)
    elif args.regularization_type == "comp2":
        Con_Loss = SupConLoss_comp_v2(temperature=args.temperature, base_temperature=args.temperature)

    elif args.regularization_type == "rr_acSCL":
        Con_Loss = SupConLoss_comp_BCE(temperature=args.temperature, base_temperature=args.temperature)

    elif args.regularization_type == "acConR":
        Con_Loss = acConR()

    elif args.regularization_type == "RNC":
        Con_Loss = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')

    elif args.regularization_type == "rank":
        Con_Loss = batchwise_ranking_regularizer
    else:
        raise "Pelease Check regularization_type"

    # evaluate only
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"===> Checkpoint '{args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing...")
        validate(args, test_loader, model, train_labels=train_labels, prefix='Test')
        return

    if args.retrain_fc:
        assert args.reweight != 'none' and args.pretrained
        print('===> Retrain last regression layer only!')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'linear' not in name:
                param.requires_grad = False

    # Loss and optimizer
    if not args.retrain_fc:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # optimize only the last linear layer
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        names = list(filter(lambda k: k is not None,
                            [k if v.requires_grad else None for k, v in model.module.named_parameters()]))
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f'===> Only optimize parameters: {names}')
        optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained}')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume) if args.gpu is None else \
                torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args.lr, [60, 80])
        train_loss = train(args, Con_Loss, train_loader, model, len(train_dataset), optimizer, epoch)
        val_loss_mse, val_loss_l1, val_loss_gmean = validate(args, val_loader, model, train_labels=train_labels)

        loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        print(f"Best {'L1' if 'l1' in args.loss else 'MSE'} Loss: {args.best_loss:.3f}")
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
              f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

    # test with best checkpoint
    print("=" * 120)
    print("Test best model on testset...")
    checkpoint = torch.load(f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")
    test_loss_mse, test_loss_l1, test_loss_gmean = validate(args, test_loader, model, train_labels=train_labels,
                                                            prefix='Test')
    print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")


def train(args, Con_Loss, train_loader, model, traindata_size, optimizer, epoch, accum_iter=1):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter(f'Loss (sum)', ':.3f')
    losses_reg = AverageMeter(f'Loss ({args.loss.upper()})', ':.3f')
    losses_con = AverageMeter(f'Loss ({"Cons"})', ':.5f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_reg, losses_con],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    n_views = args.n_views
    # accum_iter = 8
    for idx, (inputs, targets, targets_pdf, weights) in enumerate(train_loader):
        # if idx >= 10:
        #     break
        data_time.update(time.time() - end)
        targets, targets_pdf, weights = targets.cuda(non_blocking=True), \
                                        targets_pdf.cuda(non_blocking=True), \
                                        weights.cuda(non_blocking=True)
        if args.model == "contraNet_ada_queue":
            inputs = torch.cat((inputs), dim=0)
            inputs = inputs.cuda(non_blocking=True)
            outputs, features = model(inputs, targets)
            if args.loss == "bmse":
                temp_Func = BMCLoss(args.init_noise_sigma)
                loss = temp_Func(outputs, targets, weights)
            else:
                loss = globals()[f"weighted_{args.loss}_loss"](outputs, targets, weights)
            # print(loss, outputs, targets)
            losses_reg.update(loss.item(), inputs.size(0))
            if args.regularization_weight > 0:
                cons_loss = args.regularization_weight * Con_Loss(features)
                losses_con.update(cons_loss.item() / args.regularization_weight, inputs.size(0))
                loss += cons_loss
        elif args.model == "contraNet_ada_emb":
            bsz = targets.size()[0]
            inputs = torch.cat((inputs), dim=0)
            inputs = inputs.cuda(non_blocking=True)
            outputs, features = model(inputs)
            fs = torch.split(features, [bsz, ] * n_views, dim=0)
            features = torch.cat([f1.unsqueeze(1) for f1 in fs], dim=1)
            if args.loss == "bmse":

                temp_Func = BMCLoss(args.init_noise_sigma)
                loss = temp_Func(outputs, torch.cat([targets, ] * n_views, dim=0), weights)
            else:
                loss = globals()[f"weighted_{args.loss}_loss"](outputs, torch.cat([targets, ] * n_views, dim=0),
                                                           torch.cat([weights, ] * n_views, dim=0))
            losses_reg.update(loss.item(), inputs.size(0))
            if args.regularization_weight > 0:
                cons_loss = args.regularization_weight * \
                            Con_Loss(features, targets, dist=targets_pdf, norm_val=2 / traindata_size, scale_s=20)
                losses_con.update(cons_loss.item() / args.regularization_weight, inputs.size(0))
                loss += cons_loss
        elif args.regularization_type in ["ada", "rr_acSCL", "comp2", "scl"] :
            bsz = targets.size()[0]

            inputs = torch.cat((inputs), dim=0)
            inputs = inputs.cuda(non_blocking=True)
            outputs, features = model(inputs)

            fs = torch.split(features, [bsz, ] * n_views, dim=0)
            features = torch.cat([f1.unsqueeze(1) for f1 in fs], dim=1)
            if args.loss == "bmse":
                temp_Func = BMCLoss(args.init_noise_sigma)
                loss = temp_Func(outputs, targets, weights)
            else:

                loss = globals()[f"weighted_{args.loss}_loss"](outputs, torch.cat([targets, ] * n_views, dim=0),
                                                           torch.cat([weights, ] * n_views, dim=0))
            losses_reg.update(loss.item(), inputs.size(0))

            if args.regularization_weight > 0:
                cons_loss = args.regularization_weight * \
                            Con_Loss(features, targets, dist=targets_pdf, norm_val=2 / traindata_size, scale_s=20)
                losses_con.update(cons_loss.item() / args.regularization_weight, inputs.size(0))
                loss += cons_loss

        else:
            # inputs = torch.cat((inputs), dim=0)
            inputs = inputs[0].cuda(non_blocking=True)
            bsz = targets.size()[0]
            outputs, features = model(inputs)
            features = features[:bsz]
            if args.loss == "bmse":
                temp_Func = BMCLoss(args.init_noise_sigma)
                loss = temp_Func(outputs, targets, weights)
            else:
                loss = globals()[f"weighted_{args.loss}_loss"](outputs, targets, weights)
            losses_reg.update(loss.item(), inputs.size(0))
            if args.regularization_weight > 0:
                cons_loss = args.regularization_weight * \
                            Con_Loss(features, targets, dist=targets_pdf, norm_val=2 / traindata_size, scale_s=20)
                losses_con.update(cons_loss.item() / args.regularization_weight, inputs.size(0))
                loss += cons_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"
        losses.update(loss.item(), inputs.size(0))
        loss.requires_grad_(True)
        optimizer.zero_grad()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         # print(name, param.data)
        #         if param.grad is not None and torch.any(torch.isnan(param.grad)):  # Nan数据的判断方法之一
        #             print(f"Y{name}:Nan")
        loss.backward()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         # print(name, param.data)
        #         if param.grad is not None and torch.any(torch.isnan(param.grad)):  # Nan数据的判断方法之一
        #             print(f"{name}:Nan")
        # import pdb
        # pdb.set_trace()

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update:
        print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for (inputs, targets, targets_pdf, weights) in tqdm(train_loader):
                inputs = inputs.cuda(non_blocking=True)
                outputs, feature = model(inputs, )
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())

        encodings, labels = torch.from_numpy(np.vstack(encodings)) \
                                .cuda(), torch.from_numpy(np.hstack(labels)).cuda()
        model.module.FDS.update_last_epoch_stats(epoch)
        model.module.FDS.update_running_stats(encodings, labels, epoch)

    return losses.avg


def validate(args, val_loader, model, train_labels=None, prefix='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels = [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, targets_pdf, weights) in enumerate(val_loader):
            # inputs = torch.cat((inputs), dim=0)
            inputs = inputs[0].cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            try:
                outputs, _ = model(inputs)
            except:
                outputs, _ = model.module.encoder_q(inputs)
                outputs = nn.functional.relu(outputs)
            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())

            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        if args.datatype == "natural":
            shot_dict = shot_natural_metrics(np.hstack(preds), np.hstack(labels), train_labels)
            print(f" * Overall: MSE {shot_dict['all']['mse']:.3f}\t RMSE {shot_dict['all']['rmse']:.3f}\t"
                  f"L1 {shot_dict['all']['l1']:.3f}\tr2 {shot_dict['all']['r2']:.3f}")

        elif args.datatype == "balanced":
            shot_dict = shot_balanced_metrics(np.hstack(preds), np.hstack(labels), train_labels)
            print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
            print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t RMSE {shot_dict['many']['rmse']:.3f}\t"
                  f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
            print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t RMSE {shot_dict['median']['rmse']:.3f}\t"
                  f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
            print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t RMSE {shot_dict['low']['rmse']:.3f}\t"
                  f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")

    return losses_mse.avg, losses_l1.avg, loss_gmean


def shot_balanced_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, rmse_per_class, l1_per_class, l1_all_per_class = [], [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))

        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        rmse_per_class.append(np.sqrt(np.sum((preds[labels == l] - labels[labels == l]) ** 2)))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_rmse, median_shot_rmse, low_shot_rmse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []

    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_rmse.append(rmse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_rmse.append(rmse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_rmse.append(rmse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])
    print('many:{}, median:{}, few:{}'.format(np.sum(many_shot_cnt), np.sum(median_shot_cnt), np.sum(low_shot_cnt)))
    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['rmse'] = np.sum(many_shot_rmse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)

    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['rmse'] = np.sum(median_shot_rmse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)

    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['rmse'] = np.sum(low_shot_rmse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict


def shot_natural_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        y_predict = preds.detach().cpu().numpy()
        y_test = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        y_predict = preds
        y_test = labels
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')
    mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
    rmse = np.sqrt(mse)
    mae = np.sum(np.absolute(y_test - y_predict)) / len(y_test)
    r2 = 1 - mse / np.var(y_test)  # 均方误差/方差
    shot_dict = defaultdict(dict)
    shot_dict['all']['mse'] = mse
    shot_dict['all']['rmse'] = rmse
    shot_dict['all']['l1'] = mae
    shot_dict['all']['r2'] = r2
    return shot_dict


if __name__ == '__main__':
    main()
