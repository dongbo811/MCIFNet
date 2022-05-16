import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class MIF(nn.Module):
    def __init__(self):
        super(MIF, self).__init__()
        #D1
        self.conv1v = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(32)
        #L1
        self.conv2h = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(32)
        #L2
        self.conv2v = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(32)
        #last CBR
        self.conv3v = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=1)


    def forward(self, left, down):
        #left:low-level
        #down:high-level

        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        avg_out1= torch.mean(out1v, dim=1, keepdim=True)
        max_out1, _ = torch.max(out1v, dim=1, keepdim=True)

        out1h = F.relu(self.bn2h(self.conv2h(left )), inplace=True)
        avg_out2 = torch.mean(out1h, dim=1, keepdim=True)
        max_out2, _ = torch.max(out1h, dim=1, keepdim=True)

        avg_out1_1 = avg_out1 * F.interpolate(max_out2, size=avg_out1.size()[2:], mode='bilinear')
        max_out1_1 = max_out1 * F.interpolate(avg_out2, size=max_out1.size()[2:], mode='bilinear')
        avg_out2_1 = avg_out2 * F.interpolate(max_out1, size=avg_out2.size()[2:], mode='bilinear')
        max_out2_1 = max_out2 * F.interpolate(avg_out1, size=max_out2.size()[2:], mode='bilinear')

        scale1 = torch.cat([avg_out1_1, max_out1_1], dim=1)
        scale1 = self.conv1(scale1)
        scale1 = F.interpolate(scale1, size=out1v.size()[2:], mode='bilinear')
        s = out1v * self.sigmoid1(scale1) + out1v

        scale2 = torch.cat([avg_out2_1, max_out2_1], dim=1)
        scale2 = self.conv2(scale2)
        scale2 = F.interpolate(scale2, size=out1h.size()[2:], mode='bilinear')
        out1h = out1h * self.sigmoid2(scale2) + out1h
        out2v = F.relu(self.bn2v(self.conv2v(out1h)), inplace=True)

        if out2v.size()[2:] != s.size()[2:]:
            s = F.interpolate(s, size=out2v.size()[2:], mode='bilinear')

        fuse  = s*out2v
        fuse = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True)

        return fuse

    def initialize(self):
        weight_init(self)

class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_upsample44 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)

        self.mif1 = MIF()
        self.mif2 = MIF()
        self.mif3 = MIF()

    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3
        x3_12 = self.conv_upsample44(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample6(self.upsample(x3_1)) * self.conv_upsample7(
            self.upsample(x3_12)) * self.conv_upsample8(self.upsample(x3)) * x4

        # x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x1_1 = self.conv_upsample4(self.upsample(x1_1))
        x2_2 = self.mif1(x2_1, x1_1)
        x2_2 = self.conv_concat2(x2_2)

        # x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x2_2 = self.conv_upsample5(self.upsample(x2_2))
        x3_2 = self.mif2(x3_1, x2_2)
        x3_2 = self.conv_concat3(x3_2)

        # x4_2 = torch.cat((x4_1, self.conv_upsample9(self.upsample(x3_2))), 1)
        x3_2 = self.conv_upsample9(self.upsample(x3_2))
        x4_2 = self.mif3(x4_1, x3_2)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x

def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class DMC(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DMC, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv_res = BasicConv2d(in_channel, in_channel, 3)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),

        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            # BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.branch21 = nn.Sequential(BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5))
        self.branch31 = nn.Sequential( BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7))

        self.conv_cat = BasicConv2d(2*out_channel, out_channel, 3, padding=1)


    def forward(self, x):

        self.conv_res(x)

        x2 = self.branch2(x)
        x1 = self.branch3(x)+x2

        x2 = self.branch21(x1)
        x3 = self.branch31(x1)

        x_cat = self.conv_cat(torch.cat((x2, x3), 1))
        y = self.relu(x_cat)

        return y

    def initialize(self):
        weight_init(self)

class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # ---- Receptive Field Block like module ----
        self.dmc4_1 = DMC(2048, 32)
        self.dmc3_1 = DMC(1024, 32)
        self.dmc2_1 = DMC(512, 32)
        self.dmc1_1 = DMC(256, 32)

        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)

    def forward(self, x):
        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        # Receptive Field Block (enhanced)
        x1_dmc = self.dmc1_1(x1)
        x2_dmc = self.dmc2_1(x2)        # channel -> 32
        x3_dmc = self.dmc3_1(x3)        # channel -> 32
        x4_dmc = self.dmc4_1(x4)        # channel -> 32

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4_dmc, x3_dmc, x2_dmc, x1_dmc)
        S_g_pred = F.interpolate(S_g, scale_factor=4, mode='bilinear')    # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return S_g_pred
import numpy as np
import cv2
import torch

    # cv2.imshow('img', x_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
    