import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def weights_init_He_normal(m):
    classname = m.__class__.__name__
#     print classname
    if classname.find('Transpose') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Conv') != -1:
        # std = np.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels))
        # m.weight.data.normal_(0.0, std)
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()

class Detection_ResNet_BN2(nn.Module):
    def __init__(self, width):
        super(Detection_ResNet_BN2, self).__init__()
        self.conv1 = nn.Conv2d(1, width, 5, padding=2)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.PReLU(num_parameters=width)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv4 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.PReLU(num_parameters=width)
        self.conv5 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv6 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.PReLU(num_parameters=width)
        self.conv7 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv8 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.PReLU(num_parameters=width)
        self.conv9 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv9_bn = nn.BatchNorm2d(num_features=width, eps=1e-05, momentum=0.1, affine=True)
        self.conv10 = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x_res = self.conv3_bn(self.conv3(self.relu2(self.conv2_bn(self.conv2(x)))))
        x = x + x_res
        x_res = self.conv5_bn(self.conv5(self.relu4(self.conv4_bn(self.conv4(x)))))
        x = x + x_res
        x_res = self.conv7_bn(self.conv7(self.relu6(self.conv6_bn(self.conv6(x)))))
        x = x + x_res
        x_res = self.conv9_bn(self.conv9(self.relu8(self.conv8_bn(self.conv8(x)))))
        x = x + x_res
        x = self.conv10(x)
        return x

