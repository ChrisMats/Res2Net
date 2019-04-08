print("""
Unofficial implementation of the Res2Net module from:
    "Gao, S.-H., Cheng, M.-M., Zhao, K., Zhang, X.-Y., Yang, M.-H., Torr, P,
    Res2Net: A New Multi-scale Backbone Architecture,
    arXiv e-prints arXiv:1904.01169.
    https://arxiv.org/abs/1904.01169 "  
                                
Original code from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
         
Please note that this is not the official implementation!
For a proper implementation please wait for the official release from the authors
""")

import torch.nn as nn

__all__ = ['ResNet',  'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Res2Net_block(nn.Module):
    def __init__(self, planes, scale=1, stride=1, groups=1, norm_layer=None):
        super(Res2Net_block, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.scale = scale
        ch_per_sub = planes // self.scale
        ch_res = planes % self.scale    
        self.chunks  = [ch_per_sub * i + ch_res for i in range(1, scale + 1)]
        self.conv_blocks = self.get_sub_convs(ch_per_sub, norm_layer, stride, groups)
        
    def forward(self, x):
        sub_convs = []
        sub_convs.append(x[:, :self.chunks[0]])
        sub_convs.append(self.conv_blocks[0](x[:, self.chunks[0]: self.chunks[1]]))
        for s in range(2, self.scale):
            sub_x = x[:, self.chunks[s-1]: self.chunks[s]]
            sub_x += sub_convs[-1]
            sub_convs.append(self.conv_blocks[s-1](sub_x))

        return torch.cat(sub_convs, dim=1)
    
    def get_sub_convs(self, ch_per_sub, norm_layer, stride, groups):
        layers = []
        for _ in range(1, self.scale):
            layers.append(nn.Sequential(
                conv3x3(ch_per_sub, ch_per_sub, stride, groups),
                norm_layer(ch_per_sub), self.relu))
        
        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, scale=1, stride=1, downsample=None, groups=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        if downsample is None and scale > 1:
            self.conv2 = Res2Net_block(planes, scale, stride, groups)
        else:
            self.conv2 = conv3x3(planes, planes, stride, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
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

    def __init__(self, block, layers, scale=1,  num_classes=1000, zero_init_residual=False,
            groups=1,width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        self.scale = scale
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.scale, stride, downsample, groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.scale, groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(scale=1, **kwargs):
    """Constructs a Res2Net-50 model.

    Args:
        scale (int): Number of feature groups in the Res2Net block
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], scale=scale, **kwargs)
    return model


def resnet101(scale=1, **kwargs):
    """Constructs a Res2Net-101 model.

    Args:
        scale (int): Number of feature groups in the Res2Net block
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], scale=scale, **kwargs)
    return model


def resnet152(scale=1, **kwargs):
    """Constructs a Res2Net-152 model.

    Args:
        scale (int): Number of feature groups in the Res2Net block
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], scale=scale, **kwargs)
    return model


def resnext50_32x4d(scale=1, **kwargs):
    """Constructs a Res2NeXt50_32x4d model.

    Args:
        scale (int): Number of feature groups in the Res2Net block
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=4, width_per_group=32, scale=scale, **kwargs)
    return model


def resnext101_32x8d(scale=1, **kwargs):
    """Constructs a Res2NeXt101_32x8d model.

    Args:
        scale (int): Number of feature groups in the Res2Net block
        If scale=1 then it will create the standard conv3x3 block
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=8, width_per_group=32, scale=scale, **kwargs)
    return model