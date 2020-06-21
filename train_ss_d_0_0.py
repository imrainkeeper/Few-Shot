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
from models.net_ss_d_0_net0 import net
from utils.utils_ss_d import save_checkpoint
from data.data_loader_ss_d_0 import ImageDataset

parser = argparse.ArgumentParser(description='PyTorch rain24')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str, help='path to the pretrained model')
args = parser.parse_args()
args.best_precision = 0
args.best_mae = sys.maxsize
args.best_mse = sys.maxsize
args.best_epoch = -1
checkpoint_save_dir = '/home/rainkeeper/Projects/PycharmProjects/rain24/checkpoints_rain24_ss_d_0_0_try0'
terminal_log_file = os.path.join(checkpoint_save_dir, 'terminal_log.txt')
terminal_file = open(terminal_log_file, 'a')


def main():
    args.lr = 1e-4
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.alpha = 1
    args.start_epoch = 0
    args.epochs = 200
    args.workers = 2
    args.seed = time.time()
    args.print_freq = 400
    args.gpu_id = "cuda:1"
    args.dataset_id = 1

    train_image_dir = '/home/rainkeeper/Projects/Datasets/few_shot/self_supervised_data' + str(args.dataset_id) + '/Train_Image/crop_distance'
    val_image_dir = '/home/rainkeeper/Projects/Datasets/few_shot/self_supervised_data' + str(args.dataset_id) + '/Test_Image/crop_distance'

    args.device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(args.seed)

    model = net()
    model.to(args.device)

    criterion = nn.MarginRankingLoss(margin=1.0, reduction='sum').to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        print('begin train...')
        print('begin train...', file=terminal_file)
        train_accuracy = train(train_image_dir, model, criterion, optimizer, epoch)
        print('begin test...')
        print('begin test...', file=terminal_file)
        val_accuracy = val(val_image_dir, model, epoch)
        print('\n')
        print('\n', file=terminal_file)


        if epoch >= 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_save_dir, epoch, train_accuracy, val_accuracy)


def train(train_image_dir, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    trainset = ImageDataset(img_folder=train_image_dir,
                            train=True,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ])
                            )
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                                               num_workers=args.workers)
    print('epoch %d, processed %d samples, lr %.10f' % (
        epoch, epoch * len(train_loader.dataset), args.lr))
    print('epoch %d, processed %d samples, lr %.10f' % (
        epoch, epoch * len(train_loader.dataset), args.lr), file=terminal_file)

    model.train()

    count = 0
    for i, (imgs, img_name) in enumerate(train_loader):
        images = [img[1].to(args.device) for img in imgs]
        labels = [img[0].to(args.device) for img in imgs]

        distances = model(images)      # 定义: 人越小，distance越大

        x1 = torch.empty(9).to(args.device)
        x2 = torch.empty(9).to(args.device)
        y = torch.empty(9).to(args.device)
        l1 = torch.empty(9).to(args.device)
        l2 = torch.empty(9).to(args.device)

        index = 0

        for m in range(6):
            for n in range(m + 1, 6, 1):
                if labels[m] != labels[n]:
                    x1[index] = distances[m]
                    x2[index] = distances[n]
                    gt_label = torch.FloatTensor([-1]) if labels[m] < labels[n] else torch.FloatTensor([1])
                    gt_label = gt_label.to(args.device)
                    y[index] = gt_label
                    l1[index] = labels[m]
                    l2[index] = labels[n]
                    index += 1

        assert index == 9

        loss = criterion(x1, x2, y)

        losses.update(loss.item(), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.8f} ({loss.avg:.8f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses), file=terminal_file)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.8f} ({loss.avg:.8f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses))

        x1 = [x_temp.data.cpu().numpy() for x_temp in x1]
        x2 = [x_temp.data.cpu().numpy() for x_temp in x2]
        y = [y_temp.data.cpu().numpy() for y_temp in y]
        l1 = [l_temp.data.cpu().numpy() for l_temp in l1]
        l2 = [l_temp.data.cpu().numpy() for l_temp in l2]

        for p in range(len(x1)):
            if x1[p] < x2[p] and l1[p] < l2[p]:
                count += 1
            if x1[p] > x2[p] and l1[p] > l2[p]:
                count += 1

    print('count=', count)
    print('train_image_num=', len(train_loader.dataset))

    accuracy = count / (9 * len(train_loader.dataset))

    print('Epoch: [{0}][{1}]\t'
          'Train Accuracy {accuracy:.6f}\t'
        .format(
        epoch, len(train_loader.dataset), accuracy=accuracy))
    print('Epoch: [{0}][{1}]\t'
          'Train Accuracy {accuracy:.6f}\t'
        .format(
        epoch, len(train_loader.dataset), accuracy=accuracy), file=terminal_file)

    return accuracy


def val(val_image_dir, model, epoch):
    valset = ImageDataset(img_folder=val_image_dir,
                          train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                          ]),
                          )
    val_loader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=args.batch_size,
                                               num_workers=args.workers)
    print('epoch %d, processed %d samples, lr %.10f' % (
        epoch, epoch * len(val_loader.dataset), args.lr))
    print('epoch %d, processed %d samples, lr %.10f' % (
        epoch, epoch * len(val_loader.dataset), args.lr), file=terminal_file)

    model.eval()
    end = time.time()

    count = 0
    with torch.no_grad():
        for i, (imgs, img_name) in enumerate(val_loader):
            images = [img[1].to(args.device) for img in imgs]
            labels = [img[0].to(args.device) for img in imgs]

            distances = model(images)

            distances = [distance.data.cpu().numpy() for distance in distances]
            labels = [label.data.cpu().numpy() for label in labels]

            for m in range(6):
                for n in range(m + 1, 6, 1):
                    if (labels[m] - labels[n]) * (distances[m] - distances[n]) > 0:
                        count += 1

            if i % 100 == 0:
                for m in range(len(distances)):
                    print('distances: %.6f, labels: %d' % (distances[m], labels[m]))
                print('\n')
                for m in range(len(distances)):
                    print('distances: %.6f, labels: %d' % (distances[m], labels[m]), file=terminal_file)
                print('\n', file=terminal_file)

    accuracy = count / (9 * len(val_loader.dataset))

    print('count=', count)
    print('test_image_num=', len(val_loader.dataset))

    print('Epoch: [{0}][{1}]\t'
          'Test Accuracy {accuracy:.6f}\n'
        .format(
        epoch, len(val_loader.dataset), accuracy=accuracy), file=terminal_file)
    print('Epoch: [{0}][{1}]\t'
          'Test Accuracy {accuracy:.6f}\n'
        .format(
        epoch, len(val_loader.dataset), accuracy=accuracy))

    return accuracy


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