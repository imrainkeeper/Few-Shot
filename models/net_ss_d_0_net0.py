#####################################################
# rain20中net_self_supervise_crowd_count_net4_0.py 与 rain14中nets1.py 是一样的
#####################################################


import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.ss_train = nn.Sequential(nn.Conv2d(3, 32, kernel_size=9, padding=4),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2),
                                      nn.Conv2d(32, 64, kernel_size=7, padding=3),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2),
                                      nn.Conv2d(64, 128, kernel_size=7, padding=3),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2),
                                      nn.Conv2d(128, 256, kernel_size=7, padding=3),
                                      nn.ReLU(inplace=True), )
        self.fc = nn.Linear(256, 1)
        self._initialize_weights()

    def forward(self, images):
        batch_images = torch.cat((images[0], images[1], images[2], images[3], images[4], images[5]), 0)
        # print('sss', batch_images.shape)   # torch.Size([5, 3, 512, 512])

        ss_train_output = self.ss_train(batch_images)
        avg_pooling_feature = (F.avg_pool2d(ss_train_output, 64) * (64 * 64)).view(-1, 256)
        distance = self.fc(avg_pooling_feature)

        return distance

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    net = net()
    print(net)
