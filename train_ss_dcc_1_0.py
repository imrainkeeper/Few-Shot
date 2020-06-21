#################################################
# 将 distance 和 crowd count 用两个fc而不是一个fc层求值
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
from models.net_ss_dcc_1_net0 import net
from utils.utils_ss_dcc import save_checkpoint
from data.data_loader_ss_dcc_1 import ImageDataset

parser = argparse.ArgumentParser(description='PyTorch rain24')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str, help='path to the pretrained model')
args = parser.parse_args()
args.best_precision = 0
args.best_mae = sys.maxsize
args.best_mse = sys.maxsize
args.best_epoch = -1
checkpoint_save_dir = '/home/rainkeeper/Projects/PycharmProjects/rain24/checkpoints_rain24_ss_dcc_1_0_try0'
terminal_log_file = os.path.join(checkpoint_save_dir, 'terminal_log.txt')
terminal_file = open(terminal_log_file, 'a')


def main():
    args.lr = 1e-4
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.alpha = 1
    args.beta = 1
    args.start_epoch = 0
    args.epochs = 200
    args.workers = 2
    args.seed = time.time()
    args.print_freq = 400
    args.gpu_id = "cuda:4"
    args.dataset_id = 0

    train_image_dir_distance = '/home/rainkeeper/Projects/Datasets/few_shot/self_supervised_data' + str(args.dataset_id) + '/Train_Image/crop_distance'
    val_image_dir_distance = '/home/rainkeeper/Projects/Datasets/few_shot/self_supervised_data' + str(args.dataset_id) + '/Test_Image/crop_distance'
    train_image_dir_crowd_count = '/home/rainkeeper/Projects/Datasets/few_shot/self_supervised_data' + str(args.dataset_id) + '/Train_Image/crop_count'
    val_image_dir_crowd_count = '/home/rainkeeper/Projects/Datasets/few_shot/self_supervised_data' + str(args.dataset_id) + '/Test_Image/crop_count'

    args.device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(args.seed)

    model = net()
    model.to(args.device)

    criterion = nn.MarginRankingLoss(margin=1.0, reduction='sum').to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        print('begin train...')
        print('begin train...', file=terminal_file)
        train_accuracy_distance, train_accuracy_crowd_count = train(train_image_dir_distance, train_image_dir_crowd_count, model, criterion, optimizer, epoch)
        print('begin test...')
        print('begin test...', file=terminal_file)
        val_accuracy_distance, val_accuracy_crowd_count = val(val_image_dir_distance, val_image_dir_crowd_count, model, epoch)
        print('\n')
        print('\n', file=terminal_file)


        if epoch >= 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_save_dir, epoch, train_accuracy_distance, train_accuracy_crowd_count, val_accuracy_distance, val_accuracy_crowd_count)


def train(train_image_dir_distance, train_image_dir_crowd_count, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    losses_d = AverageMeter()
    losses_cc = AverageMeter()

    trainset = ImageDataset(img_folder_d=train_image_dir_distance,
                            img_folder_cc=train_image_dir_crowd_count,
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

    count_d = 0
    count_cc = 0
    for i, (imgs, img_name) in enumerate(train_loader):
        images = [img[1].to(args.device) for img in imgs]
        labels = [img[0].to(args.device) for img in imgs]

        distance_and_count = model(images)      # 定义: 人越远，distance越大,则label越大

        x1_d = torch.empty(9).to(args.device)
        x2_d = torch.empty(9).to(args.device)
        y_d = torch.empty(9).to(args.device)
        l1_d = torch.empty(9).to(args.device)
        l2_d = torch.empty(9).to(args.device)

        x1_cc = torch.empty(10).to(args.device)
        x2_cc = torch.empty(10).to(args.device)
        y_cc = torch.empty(10).to(args.device)
        l1_cc = torch.empty(10).to(args.device)
        l2_cc = torch.empty(10).to(args.device)

        anchor = torch.ones(1).to(args.device)

        index_d = 0
        index_cc = 0

        for m in range(11):
            for n in range(m + 1, 11, 1):
                if labels[m] <= anchor and labels[n] <= anchor:
                    if labels[m] != labels[n]:
                        x1_d[index_d] = distance_and_count[m]
                        x2_d[index_d] = distance_and_count[n]
                        gt_label = torch.FloatTensor([-1]) if labels[m] < labels[n] else torch.FloatTensor([1])
                        gt_label = gt_label.to(args.device)
                        y_d[index_d] = gt_label
                        l1_d[index_d] = labels[m]
                        l2_d[index_d] = labels[n]
                        index_d += 1
                elif labels[m] > anchor and labels[n] > anchor:
                    x1_cc[index_cc] = distance_and_count[m]
                    x2_cc[index_cc] = distance_and_count[n]
                    gt_label = torch.FloatTensor([-1]) if labels[m] < labels[n] else torch.FloatTensor([1])
                    gt_label = gt_label.to(args.device)
                    y_cc[index_cc] = gt_label
                    l1_cc[index_cc] = labels[m]
                    l2_cc[index_cc] = labels[n]
                    index_cc += 1
                else:
                    pass

        assert index_d == 9 and index_cc == 10

        loss_d = criterion(x1_d, x2_d, y_d)
        loss_cc = criterion(x1_cc, x2_cc, y_cc)
        loss = args.alpha * loss_d + args.beta * loss_cc

        losses_d.update(loss_d.item(), 1)
        losses_cc.update(loss_cc.item(), 1)
        losses.update(loss.item(), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.8f} ({loss.avg:.8f})\t'
                  'Loss_d {loss_d.val:.8f} ({loss_d.avg:.8f})\t'
                  'Loss_cc {loss_cc.val:.8f} ({loss_cc.avg:.8f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses, loss_d=losses_d, loss_cc=losses_cc),
                file=terminal_file)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.8f} ({loss.avg:.8f})\t'
                  'Loss_d {loss_d.val:.8f} ({loss_d.avg:.8f})\t'
                  'Loss_cc {loss_cc.val:.8f} ({loss_cc.avg:.8f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses, loss_d=losses_d, loss_cc=losses_cc))

        x1_d = [x_temp.data.cpu().numpy() for x_temp in x1_d]
        x2_d = [x_temp.data.cpu().numpy() for x_temp in x2_d]
        l1_d = [l_temp.data.cpu().numpy() for l_temp in l1_d]
        l2_d = [l_temp.data.cpu().numpy() for l_temp in l2_d]

        for p in range(len(x1_d)):
            if x1_d[p] < x2_d[p] and l1_d[p] < l2_d[p]:
                count_d += 1
            if x1_d[p] > x2_d[p] and l1_d[p] > l2_d[p]:
                count_d += 1

        x1_cc = [x_temp.data.cpu().numpy() for x_temp in x1_cc]
        x2_cc = [x_temp.data.cpu().numpy() for x_temp in x2_cc]
        l1_cc = [l_temp.data.cpu().numpy() for l_temp in l1_cc]
        l2_cc = [l_temp.data.cpu().numpy() for l_temp in l2_cc]

        for p in range(len(x1_cc)):
            if x1_cc[p] < x2_cc[p] and l1_cc[p] < l2_cc[p]:
                count_cc += 1
            if x1_cc[p] > x2_cc[p] and l1_cc[p] > l2_cc[p]:
                count_cc += 1

    print('count_distance=', count_d)
    print('count_crowd_count=', count_cc)
    print('train_image_num=', len(train_loader.dataset))

    accuracy_distance = count_d / (9 * len(train_loader.dataset))
    accuracy_crowd_count = count_cc / (10 * len(train_loader.dataset))

    print('Epoch: [{0}][{1}]\t'
          'Accuracy_d {accuracy_d:.6f}\t'
          'Accuracy_cc {accuracy_cc:.6f}\t'
        .format(
        epoch, len(train_loader.dataset), accuracy_d=accuracy_distance, accuracy_cc=accuracy_crowd_count),
        file=terminal_file)
    print('Epoch: [{0}][{1}]\t'
          'Accuracy_d {accuracy_d:.6f}\t'
          'Accuracy_cc {accuracy_cc:.6f}\t'
        .format(
        epoch, len(train_loader.dataset), accuracy_d=accuracy_distance, accuracy_cc=accuracy_crowd_count))

    return accuracy_distance, accuracy_crowd_count


def val(val_image_dir_distance, val_image_dir_crowd_count, model, epoch):
    valset = ImageDataset(img_folder_d=val_image_dir_distance,
                          img_folder_cc=val_image_dir_crowd_count,
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

    count_d = 0
    count_cc = 0

    with torch.no_grad():
        for i, (imgs, img_name) in enumerate(val_loader):
            images = [img[1].to(args.device) for img in imgs]
            labels = [img[0].to(args.device) for img in imgs]

            distance_and_count = model(images)

            distance = [d.data.cpu().numpy() for d in distance_and_count[:6]]
            count = [cc.data.cpu().numpy() for cc in distance_and_count[6:11]]
            labels = [label.data.cpu().numpy() for label in labels]

            for m in range(6):
                for n in range(m + 1, 6, 1):
                    if (labels[m] - labels[n]) * (distance[m] - distance[n]) > 0:
                        count_d += 1

            for m in range(5):
                for n in range(m + 1, 5, 1):
                    if (labels[m + 6] - labels[n + 6]) * (count[m] - count[n]) > 0:
                        count_cc += 1

            if i % 100 == 0:
                for m in range(len(distance)):
                    print('distances: %.6f' % distance[m])
                print('\n')
                for n in range(len(count)):
                    print('crowd counts: %.6f' % count[n])
                print('\n')
                for m in range(len(distance)):
                    print('distances: %.6f' % distance[m], file=terminal_file)
                print('\n', file=terminal_file)
                for n in range(len(count)):
                    print('crowd counts: %.6f' % count[n], file=terminal_file)
                print('\n', file=terminal_file)

        accuracy_distance = count_d / (9 * len(val_loader.dataset))
        accuracy_crowd_count = count_cc / (10 * len(val_loader.dataset))

        print('count_distance=', count_d)
        print('count_crowd_count=', count_cc)
        print('test_image_num=', len(val_loader.dataset))

        print('Epoch: [{0}][{1}]\t'
              'Accuracy_d {accuracy_d:.6f}\t'
              'Accuracy_cc {accuracy_cc:.6f}\t'
            .format(
            epoch, len(val_loader.dataset), accuracy_d=accuracy_distance, accuracy_cc=accuracy_crowd_count),
            file=terminal_file)
        print('Epoch: [{0}][{1}]\t'
              'Accuracy_d {accuracy_d:.6f}\t'
              'Accuracy_cc {accuracy_cc:.6f}\t'
            .format(
            epoch, len(val_loader.dataset), accuracy_d=accuracy_distance, accuracy_cc=accuracy_crowd_count))

        return accuracy_distance, accuracy_crowd_count


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
