#####################################################
# rain20中的net_full_supervise_crowd_count_net4_0.py 与 rain14中的 netf2.py 相同
#####################################################


import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, model_path):
        super(net, self).__init__()
        self.model_path = model_path

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

        self.fs_train = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 1, kernel_size=1))

        self._initialize_weights()

        checkpoint = torch.load(self.model_path)
        for i in range(len(self.ss_train.state_dict().items())):
            list(self.ss_train.state_dict().items())[i][1].data[:] = list(checkpoint['state_dict'].items())[i][1].data[:]

    def forward(self, image):
        ss_train_output = self.ss_train(image)
        predict_density_map = self.fs_train(ss_train_output)
        return predict_density_map
        # return middle_feature

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
