#################################################
#
#################################################

import sys
import os
import warnings
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
import cv2
import time
from models.net_fs_net0 import net
from utils.utils_fs import save_checkpoint
from data.data_loader_fs_0 import ImageDataset

parser = argparse.ArgumentParser(description='PyTorch rain24')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str, help='path to the pretrained model')
args = parser.parse_args()
args.best_precision = 0
args.best_mae = sys.maxsize
args.best_mse = sys.maxsize
args.best_epoch = -1
checkpoint_save_dir = '/home/rainkeeper/Projects/PycharmProjects/rain24/checkpoints_rain24_fs_cc_0_0_3p_try8'
terminal_log_file = os.path.join(checkpoint_save_dir, 'terminal_log.txt')
terminal_file = open(terminal_log_file, 'a')


def main():
    args.lr = 1e-4
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.alpha = 1
    args.start_epoch = 0
    args.epochs = 1000
    args.workers = 1
    args.seed = time.time()
    args.print_freq = 400
    args.gpu_id = "cuda:5"
    args.dataset = 'A'
    args.dataset_id = 0

    args.model_path = '/home/rainkeeper/Projects/PycharmProjects/rain24/checkpoints_rain24_ss_cc_0_0_try0/checkpoint_44_0.9901481888035126_0.9901197604790419.pth.tar'
    train_image_dir = '/home/rainkeeper/Projects/Datasets/few_shot/shanghaiTech_few_shot' + str(args.dataset_id) + '/part_' + args.dataset + '/train_image3'
    train_gt_dir = '/home/rainkeeper/Projects/Datasets/few_shot/shanghaiTech_few_shot' + str(args.dataset_id) + '/part_' + args.dataset + '/train_gt3'
    test_image_dir = '/home/rainkeeper/Projects/Datasets/few_shot/shanghaiTech_few_shot' + str(args.dataset_id) + '/part_' + args.dataset + '/test_image'
    test_gt_dir = '/home/rainkeeper/Projects/Datasets/few_shot/shanghaiTech_few_shot' + str(args.dataset_id) + '/part_' + args.dataset + '/test_gt'

    args.device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(args.seed)

    model = net(args.model_path)
    model.to(args.device)

    criterion = nn.MSELoss(reduction='sum').to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # if args.pre:
    if False:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            args.best_mae = checkpoint['best_mae']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        train_mae, train_mse, train_gt_sum, train_predict_sum = train(train_image_dir, train_gt_dir, model, criterion, optimizer, epoch)

        print(
            'current train mae: %.6f, current train mse: %.6f, current_train_gt_sum: %.6f, current_train_predict_sum:%.6f' % (
                train_mae, train_mse, train_gt_sum, train_predict_sum))
        print(
            'current train mae: %.6f, current train mse: %.6f, current_train_gt_sum: %.6f, current_train_predict_sum:%.6f' % (
                train_mae, train_mse, train_gt_sum, train_predict_sum), file=terminal_file)

        if epoch >= 0:
            test_mae, test_mse, test_gt_sum, test_predict_sum = sssss(test_image_dir, test_gt_dir, model)

            is_best = (test_mae <= args.best_mae)
            if is_best:
                args.best_mae = test_mae
                args.best_mse = test_mse
                args.best_epoch = epoch

            print(
                'current test mae: %.6f, current test mse: %.6f, current_test_gt_sum: %.6f, current_test_predict_sum:%.6f' % (
                    test_mae, test_mse, test_gt_sum, test_predict_sum))
            print('best test mae: %.6f, best test mse: %.6f' % (args.best_mae, args.best_mse))
            print('best epoch:%d' % args.best_epoch)
            print('\n')
            print(
                'current test mae: %.6f, current test mse: %.6f, current_test_gt_sum: %.6f, current_test_predict_sum:%.6f' % (
                    test_mae, test_mse, test_gt_sum, test_predict_sum), file=terminal_file)
            print('best test mae: %.6f, best test mse: %.6f' % (args.best_mae, args.best_mse), file=terminal_file)
            print('best epoch:%d' % args.best_epoch, file=terminal_file)
            print('\n', file=terminal_file)

            if is_best and test_mae < 280:
                test_mae_2f = round(test_mae, 2)
                test_mse_2f = round(test_mse, 2)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.pre,
                    'state_dict': model.state_dict(),
                    'current_mae': test_mae,
                    'best_mae': args.best_mae,
                    'optimizer': optimizer.state_dict()
                }, checkpoint_save_dir, args.dataset, epoch, test_mae_2f, test_mse_2f)


def train(train_image_dir, train_gt_dir, model, criterion, optimizer, epoch):
    print('begin train...')
    print('begin train...', file=terminal_file)

    losses = AverageMeter()

    trainset = ImageDataset(img_dir=train_image_dir,
                            gt_dir=train_gt_dir,
                            train=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            )
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                                               num_workers=args.workers)
    print('epoch %d, processed %d samples, dataset %s, lr %.10f' % (
        epoch, epoch * len(train_loader.dataset), args.dataset, args.lr))
    print('epoch %d, processed %d samples, dataset %s, lr %.10f' % (
        epoch, epoch * len(train_loader.dataset), args.dataset, args.lr), file=terminal_file)

    model.train()
    end = time.time()

    train_mae = 0.0
    train_mse = 0.0
    train_gt_sum = 0.0
    train_predict_sum = 0.0
    for i, (img, gt_density_map) in enumerate(train_loader):
        img = img.to(args.device)
        gt_density_map = gt_density_map.to(args.device)

        predict_density_map = model(img)  #

        loss = criterion(predict_density_map, gt_density_map)

        losses.update(loss.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses), file=terminal_file)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses))

        train_gt_count = np.sum(gt_density_map.data.cpu().numpy())
        train_predict_count = np.sum(predict_density_map.data.cpu().numpy())
        train_mae += abs(train_gt_count - train_predict_count)
        train_mse += (train_gt_count - train_predict_count) * (train_gt_count - train_predict_count)
        train_gt_sum += train_gt_count
        train_predict_sum += train_predict_count
    train_mae = train_mae / len(train_loader.dataset)
    train_mse = np.sqrt(train_mse / len(train_loader.dataset))
    train_gt_sum = train_gt_sum / len(train_loader.dataset)
    train_predict_sum = train_predict_sum / len(train_loader.dataset)

    return train_mae, train_mse, train_gt_sum, train_predict_sum


# def validate(val_image_dir, val_gt_dir, model):
#     print('begin validate')
#     print('begin validate', file=terminal_file)
#     valset = ImageDataset(img_dir=val_image_dir,
#                           gt_dir=val_gt_dir,
#                           train=False,
#                           transform=transforms.Compose([
#                               transforms.ToTensor(),
#                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                    std=[0.229, 0.224, 0.225])
#                           ]),
#                           )
#     val_loader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=args.batch_size,
#                                              num_workers=args.workers)
#     model.eval()
#
#     mae = 0
#     mse = 0
#     gt_sum = 0
#     predict_sum = 0
#     for i, (img, gt_density_map, gt_label) in enumerate(val_loader):
#         img = img.to(args.device)
#         gt_density_map = gt_density_map.to(args.device)
#         predict_density_map, refined_density_map = model(img, gt_density_map)
#
#         gt_count = np.sum(gt_density_map.data.cpu().numpy())
#         predict_count = np.sum(predict_density_map.data.cpu().numpy())
#         mae += abs(gt_count - predict_count)
#         mse += ((gt_count - predict_count) * (gt_count - predict_count))
#         gt_sum += gt_count
#         predict_sum += predict_count
#     mae = mae / len(val_loader.dataset)
#     mse = np.sqrt(mse / len(val_loader.dataset))
#     gt_sum = gt_sum / len(val_loader.dataset)
#     predict_sum = predict_sum / len(val_loader.dataset)
#
#     return mae, mse, gt_sum, predict_sum


def sssss(test_image_dir, test_gt_dir, model):
    print('begin test')
    print('begin test', file=terminal_file)

    testset = ImageDataset(img_dir=test_image_dir,
                           gt_dir=test_gt_dir,
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]),
                           )
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=args.batch_size,
                                              num_workers=args.workers)
    model.eval()

    mae = 0
    mse = 0
    gt_sum = 0
    predict_sum = 0
    with torch.no_grad():
        for i, (img, gt_density_map) in enumerate(test_loader):
            img = img.to(args.device)
            gt_density_map = gt_density_map.to(args.device)
            predict_density_map = model(img)

            gt_count = np.sum(gt_density_map.data.cpu().numpy())
            predict_count = np.sum(predict_density_map.data.cpu().numpy())
            mae += abs(gt_count - predict_count)
            mse += ((gt_count - predict_count) * (gt_count - predict_count))
            gt_sum += gt_count
            predict_sum += predict_count
    mae = mae / len(test_loader.dataset)
    mse = np.sqrt(mse / len(test_loader.dataset))
    gt_sum = gt_sum / len(test_loader.dataset)
    predict_sum = predict_sum / len(test_loader.dataset)

    return mae, mse, gt_sum, predict_sum


def adjust_learning_rate(optimizer, epoch):
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()