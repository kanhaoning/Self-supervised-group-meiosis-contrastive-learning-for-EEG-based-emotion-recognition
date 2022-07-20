import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x15(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 15), stride=stride,
                     padding=(0, 7), groups=groups, bias=False, dilation=dilation)

def conv1x13(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 13), stride=stride,
                     padding=(0, 6), groups=groups, bias=False, dilation=dilation)

def conv1x11(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 11), stride=stride,
                     padding=(0, 5), groups=groups, bias=False, dilation=dilation)

def conv1x9(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 9), stride=stride,
                     padding=(0, 4), groups=groups, bias=False, dilation=dilation)

def conv1x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 7), stride=stride,
                     padding=(0, 3), groups=groups, bias=False, dilation=dilation)

def conv1x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 5), stride=stride,
                     padding=(0, 2), groups=groups, bias=False, dilation=dilation)

def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride,
                     padding=(0, 1), groups=groups, bias=False, dilation=dilation)

def conv15x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(15, 1), stride=stride,
                     padding=(7, 0), groups=groups, bias=False, dilation=dilation)

def conv13x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(13, 1), stride=stride,
                     padding=(6, 0), groups=groups, bias=False, dilation=dilation)

def conv11x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(11, 1), stride=stride,
                     padding=(5, 0), groups=groups, bias=False, dilation=dilation)

def conv9x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(9, 1), stride=stride,
                     padding=(4, 0), groups=groups, bias=False, dilation=dilation)

def conv7x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(7, 1), stride=stride,
                     padding=(3, 0), groups=groups, bias=False, dilation=dilation)

def conv5x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(5, 1), stride=stride,
                     padding=(2, 0), groups=groups, bias=False, dilation=dilation)

def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=stride,
                     padding=(1, 0), groups=groups, bias=False, dilation=dilation)

def conv17x17(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """17x17 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=17, stride=stride,
                     padding=8, groups=groups, bias=False, dilation=dilation)

def conv15x15(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """15x15 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=15, stride=stride,
                     padding=7, groups=groups, bias=False, dilation=dilation)

def conv13x13(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """13x13 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=13, stride=stride,
                     padding=6, groups=groups, bias=False, dilation=dilation)

def conv11x11(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """11x11 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=11, stride=stride,
                     padding=5, groups=groups, bias=False, dilation=dilation)

def conv9x9(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """9x9 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=9, stride=stride,
                     padding=4, groups=groups, bias=False, dilation=dilation)

def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, i, Layer, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if Layer == "1":
            if i == 1:
                self.conv1 = conv1x15(inplanes, planes, stride=(1, 2))
                self.conv2 = conv15x1(planes, planes, stride=(2, 1))
            else:
                self.conv1 = conv1x15(inplanes, planes, stride=(1, 1))
                self.conv2 = conv15x1(planes, planes)
        elif Layer == "2":
            if i == 1:
                self.conv1 = conv1x11(inplanes, planes, stride=(1, 2))
                self.conv2 = conv11x1(planes, planes, stride=(2, 1))
            else:
                self.conv1 = conv1x11(inplanes, planes, stride=(1, 1))
                self.conv2 = conv11x1(planes, planes)
        elif Layer == "3":
            if i == 1:
                self.conv1 = conv1x7(inplanes, planes, stride=(1, 2))
            else:
                self.conv1 = conv7x1(inplanes, planes, stride=(1, 1))
            self.conv2 = conv3x1(planes, planes)
        elif Layer == "4":
            if i == 1:
                self.conv1 = conv1x3(inplanes, planes, stride=(1, 2))
            else:
                self.conv1 = conv1x3(inplanes, planes, stride=(1, 1))
            self.conv2 = conv5x5(planes, planes)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=4,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self._arch = 'resnet18'
        self.layers = [2, 2, 2, 2]
        self.block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4),
                               bias=False)
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(5,1), stride=(2,1), padding=(2,0),
        #                        bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        # self.maxpool = nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.layer1 = self._make_layer("1", self.block, 64, self.layers[0])
        self.layer2 = self._make_layer("2", self.block, 128, self.layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer("3", self.block, 256, self.layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer("4", self.block, 512, self.layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_classifier1 = nn.Linear(512 * self.block.expansion, 256 * self.block.expansion)
        self.bn_fc1 = nn.BatchNorm1d(256 * self.block.expansion)

        self.fc_classifier2 = nn.Linear(256 * self.block.expansion, 128 * self.block.expansion)
        self.bn_fc2 = nn.BatchNorm1d(128 * self.block.expansion)

        self.fc_classifier3 = nn.Linear(128 * self.block.expansion, num_classes)
        self.bn_fc_p1 = nn.BatchNorm1d(1024 * self.block.expansion)
        self.fc_projector1 = nn.Linear(512 * self.block.expansion, 1024 * self.block.expansion)
        self.bn_fc_p1 = nn.BatchNorm1d(1024)
        self.fc_projector2 = nn.Linear(1024 * self.block.expansion, 2048 * self.block.expansion)
        self.bn_fc_p2 = nn.BatchNorm1d(2048 * self.block.expansion)

        self.fc_projector3 = nn.Linear(2048 * self.block.expansion, 4096 * self.block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, Layer, block, planes, blocks, stride=1, dilate=False):
        #        self.layer1 = self._make_layer("1", block, 64, layers[0])

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if Layer == "1":
            stride = (2, 2)
        elif Layer == "2":
            stride = (2, 2)
        elif Layer == "3":
            stride = (1, 2)
        elif Layer == "4":
            stride = (1, 2)
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
        #                     self.base_width, previous_dilation, norm_layer))
        # self.inplanes = planes * block.expansion
        # for _ in range(1, blocks):
        for i in range(1, blocks + 1):
            if i == 1:
                layers.append(block(i, Layer, self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer))
                self.inplanes = planes * block.expansion
            else:
                layers.append(block(i, Layer, self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, mode='embedding'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if mode == 'classifier':
            x = self.fc_classifier1(x)
            x = self.bn_fc1(x)
            x = self.relu(x)
            x = self.fc_classifier2(x)
            x = self.bn_fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc_classifier3(x)
            return x
        elif mode == 'contrast':
            x = self.fc_projector1(x)
            x = self.bn_fc_p1(x)
            x = F.dropout(x, p=0.2)
            x = self.fc_projector2(x)
            x = self.bn_fc_p2(x)
            x = F.dropout(x, p=0.2)
            x = self.fc_projector3(x)
            return x
        elif mode == 'embedding':
            return x
