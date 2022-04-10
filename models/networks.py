import imp
from turtle import forward
import torch
from torch import nn
import torchvision.models.resnet as resnet
import math
from modules.modulated_deform_conv import ModulatedDeformConvPack

class DeconvLayer(nn.Module):
    
    def __init__(
        self, in_planes,
        out_planes, deconv_kernel,
        deconv_stride=2, deconv_pad=1,
        deconv_out_pad=0, modulate_deform=True,
    ):
        super(DeconvLayer, self).__init__()
        if modulate_deform:
            self.dcn = ModulatedDeformConvPack(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
                stride=1, padding=1,
            )
        else:
            self.dcn = ModulatedDeformConvPack(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
                stride=1, padding=1,
            )

        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]


class CenternetDeconv(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self):
        super(CenternetDeconv, self).__init__()
        # modify into config
        channels = [512, 256, 128, 64]
        deconv_kernel = [4, 4, 4]
        modulate_deform = True
        self.deconv1 = DeconvLayer(
            channels[0], channels[1],
            deconv_kernel=deconv_kernel[0],
            modulate_deform=modulate_deform,
        )
        self.deconv2 = DeconvLayer(
            channels[1], channels[2],
            deconv_kernel=deconv_kernel[1],
            modulate_deform=modulate_deform,
        )
        self.deconv3 = DeconvLayer(
            channels[2], channels[3],
            deconv_kernel=deconv_kernel[2],
            modulate_deform=modulate_deform,
        )

    def forward(self, x):
        x = self.deconv1(x)  # *2
        x = self.deconv2(x)  # *2
        x = self.deconv3(x)  # *2
        return x

_resnet_mapper = {
    18: resnet.resnet18,
    50: resnet.resnet50,
    101: resnet.resnet101,
}


class ResnetBackbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        depth = 18
        backbone = _resnet_mapper[depth](pretrained=pretrained)
        self.stage0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        x = self.stage0(x)  # //4
        x = self.stage1(x)  # //1
        x = self.stage2(x)  # //2
        x = self.stage3(x)  # //2
        x = self.stage4(x)  # //2
        return x

class SingleHead(nn.Module):
    
    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self):
        super(CenternetHead, self).__init__()
        self.cls_head = SingleHead(
            64,
            1,  # holes without classes
            bias_fill=True,
            bias_value=-2.19,
        )
        self.reg_head = SingleHead(64, 2)

    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)
        reg = self.reg_head(x)
        pred = {
            'cls': cls,
            'reg': reg
        }
        return pred

class CenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResnetBackbone()
        self.deconv = CenternetDeconv()
        self.head = CenternetHead()

    def forward(self, x):
        x = self.backbone(x)  # B 512, H/16, W/16
        x = self.deconv(x)  # B 64, H/4, W/4
        x = self.head(x)
        return x
