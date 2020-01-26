
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from layers.modules.l2norm import L2Norm
from data import *
import os
import math

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes,resnet18):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = VOC_300_2
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        # self.vgg = nn.ModuleList(base)

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(256, 20)
        self.L2Norm2 = L2Norm(512, 20)

        self.vgg1 = nn.ModuleList(base[0])
        self.vgg2 = nn.ModuleList(base[1])
        #self.vgg3 = nn.ModuleList(base[2])
        #self.vgg4 = nn.ModuleList(base[3])
        self.vgg5 = nn.ModuleList(base[4])
        self.vgg6 = nn.ModuleList(base[5])
        self.vgg7 = nn.ModuleList(base[6])
        self.vgg8 = nn.ModuleList(base[7])
        self.de1 = nn.ModuleList(base[8])
        self.de2 = nn.ModuleList(base[9])
        self.de3 = nn.ModuleList(base[10])
        self.de4 = nn.ModuleList(base[11])

        self.d19sample1 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.d19sample2 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.d19sample3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=2, stride=4, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.ds38_19 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.ds19_10 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.ds10_5 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.ds5_3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        '''
        self.de5_19 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=4, padding=0, output_padding=0),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))
        
        self.de10_38 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        '''
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.con_press38= nn.Sequential(nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                                      nn.BatchNorm2d(128))
        '''
        self.con_press19 = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1)),
                                         nn.BatchNorm2d(128))
        self.con_press10 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                                         nn.BatchNorm2d(128))
        '''
        if phase == 'test':
            self.softmax = nn.Softmax()
            #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        '''
        res=x
        x = self.layer3[0].conv1(x)
        x = self.layer3[0].bn1(x)
        x = self.layer3[0].relu(x)
        x = self.layer3[0].conv2(x)

        x = self.layer3[0].bn2(x)
        res= self.layer3[0].downsample(res)
        x=x+res
        x=self.layer3[0].relu(x)
        x = self.layer3[1](x)
        '''
        x = self.layer3(x)
        res38 = x

        s = self.L2Norm(res38)
        s2 = s
        for k in range(len(self.vgg2)):
            s2 = self.vgg2[k](s2)
        #s4 = s
        #for k in range(len(self.vgg3)):
            #s4 = self.vgg3[k](s4)

        #s6 = s
        #for k in range(len(self.vgg4)):
            #s6 = self.vgg4[k](s6)

        s8 = s
        for k in range(len(self.vgg5)):
            s8 = self.vgg5[k](s8)

        for k in range(len(self.vgg6)):
            s = self.vgg6[k](s)

        s = torch.cat((s, s2, s8), 1)
        for k in range(len(self.vgg7)):
            s = self.vgg7[k](s)
        s38 = self.L2Norm2(s)
        # sources.append(s)
        ds19 = self.ds38_19(s38)

        x = self.layer4(x)

        # apply vgg up to fc7
        for k in range(len(self.vgg1)):
            x = self.vgg1[k](x)

            # if (k == 2):
            # x=x*0.5+res19*0.5
        ds10 = self.ds19_10(x)
        xde38 = x
        for k in range(len(self.de4)):
            xde38 = self.de4[k](xde38)

        s38_1=self.con_press38(s38)
        # sources.append(s38)

        x19 = self.extras[21](x)

        s19 = self.extras[22](x19)

        # sources.append(x19)
        res10 = self.d19sample1(x)
        res5 = self.d19sample2(x)
        res3 = self.d19sample3(x)
        feamp = [res10, res5, res3]
        # apply extra layers and cache source layer outputs
        for k in range(len(self.extras)):
            if (k == 21):
                break
            x = self.extras[k](x)

            if (k == 6):
                #s38_2 = self.de10_38(x)
                #s38_2=s38_1+s38_2
                s38=torch.cat((s38,s38_1,xde38),1)
                for k in range(len(self.vgg8)):
                    s38 = self.vgg8[k](s38)

                sources.append(s38)
                ds5 = self.ds10_5(x)
                xde19 = x
                for k in range(len(self.de3)):
                    xde19 = self.de3[k](xde19)
                xde19 = ds19 + xde19
                s19 = torch.cat((s19, ds19, xde19), 1)
                s19 = self.extras[23](s19)
                s19 = self.extras[24](s19)
                s19 = self.extras[25](s19)
                sources.append(s19)

                s10=x

                # sources.append(x10)
            elif (k == 13):
                #s19_2 = self.de5_19(x)


                #s19 = s19 + s19_2
                s5 = x
                ds3 = self.ds5_3(x)
                xde10 = x

                for k in range(len(self.de2)):
                    xde10 = self.de2[k](xde10)
                xde10=xde10+ds10
                s10 = torch.cat((s10, ds10,xde10), 1)
                x10 = self.extras[26](s10)
                s10 = self.extras[27](x10)
                s10 = self.extras[28](s10)
                #s10 = s10 + xde10

                sources.append(s10)


                # sources.append(x5)
            elif (k == 20):

                xde5 = x
                for k in range(len(self.de1)):
                    xde5 = self.de1[k](xde5)
                xde5=xde5+ds5
                s5 = torch.cat((s5, ds5,xde5), 1)

                x5 = self.extras[29](s5)
                s5 = self.extras[30](x5)
                s5 = self.extras[31](s5)
                sources.append(s5)

                s3 = torch.cat((x, ds3), 1)
                x3 = self.extras[32](s3)

                s3 = self.extras[33](x3)

                s3 = self.extras[34](s3)

                sources.append(s3)

            if (k == 0):
                x = torch.cat((x, res10), 1)
            elif (k == 7):
                x = torch.cat((x, res5), 1)
            elif (k == 14):
                x = torch.cat((x, res3), 1)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2)
    conv8_4 = nn.Conv2d(256, 512, kernel_size=3, padding=4, dilation=4)
    conv8_6 = nn.Conv2d(256, 512, kernel_size=3, padding=6, dilation=6)
    conv8_8 = nn.Conv2d(256, 512, kernel_size=3, padding=8, dilation=8)

    conv9_2 = nn.Conv2d(512, 512, kernel_size=1)
    conv9_4 = nn.Conv2d(512, 512, kernel_size=1)
    conv9_6 = nn.Conv2d(512, 512, kernel_size=1)
    conv9_8 = nn.Conv2d(512, 512, kernel_size=1)

    conv9_2_ = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=2)
    conv9_4_ = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=2)
    conv9_6_ = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=2)
    conv9_8_ = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=2)

    conv10_2 = nn.Conv2d(512, 128, kernel_size=1)
    conv10_4 = nn.Conv2d(512, 128, kernel_size=1)
    conv10_6 = nn.Conv2d(512, 128, kernel_size=1)
    conv10_8 = nn.Conv2d(512, 128, kernel_size=1)

    conv11 = nn.Conv2d(256, 1024, kernel_size=1)
    conv12 = nn.Conv2d(1280, 512, kernel_size=1)
    conv13 = nn.Conv2d(768, 512, kernel_size=1)

    de3_5 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=0)
    de3_5_0 = nn.BatchNorm2d(512)
    de3_5_1 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    de3_5_2 = nn.BatchNorm2d(128)
    de3_5_3 = nn.ReLU(inplace=True)

    de5_10 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
    de5_10_0 = nn.BatchNorm2d(512)
    de5_10_1 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    de5_10_2 = nn.BatchNorm2d(128)
    de5_10_3 = nn.ReLU(inplace=True)

    de10_19 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=0)
    de10_19_0 = nn.BatchNorm2d(512)
    de10_19_1 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    de10_19_2 = nn.BatchNorm2d(128)
    de10_19_3 = nn.ReLU(inplace=True)

    de19_38 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
    de19_38_0 = nn.BatchNorm2d(512)
    de19_38_1 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    de19_38_2 = nn.BatchNorm2d(128)
    de19_38_3 = nn.ReLU(inplace=True)

    layers += [pool5, conv6, nn.BatchNorm2d(1024),
               nn.ReLU(inplace=True), conv7, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
    layer1 = layers
    layer21 = [conv8_2, nn.BatchNorm2d(512), nn.ReLU(inplace=True), conv9_2, nn.BatchNorm2d(512), nn.ReLU(inplace=True),
               conv9_2_, nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                conv10_2, nn.BatchNorm2d(128), nn.ReLU(inplace=True)]

    layer22 = [conv8_4, nn.BatchNorm2d(512), nn.ReLU(inplace=True), conv9_4, nn.BatchNorm2d(512), nn.ReLU(inplace=True),
               conv9_4_, nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                conv10_4, nn.BatchNorm2d(128), nn.ReLU(inplace=True)]

    layer23 = [conv8_6, nn.BatchNorm2d(512), nn.ReLU(inplace=True), conv9_6, nn.BatchNorm2d(512), nn.ReLU(inplace=True),
               conv9_6_, nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                conv10_6, nn.BatchNorm2d(128), nn.ReLU(inplace=True)]

    layer24 = [conv8_8, nn.BatchNorm2d(512), nn.ReLU(inplace=True), conv9_8, nn.BatchNorm2d(512), nn.ReLU(inplace=True),
               conv9_8_, nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                conv10_8, nn.BatchNorm2d(128), nn.ReLU(inplace=True)]

    layer25 = [conv11, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
    layer26 = [conv12, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    layer27 = [conv13, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]

    layer3 = [de3_5, de3_5_0, de3_5_1, de3_5_2, de3_5_3]
    layer4 = [de5_10, de5_10_0, de5_10_1, de5_10_2, de5_10_3]
    layer5 = [de10_19, de10_19_0, de10_19_1, de10_19_2, de10_19_3]
    layer6 = [de19_38, de19_38_0, de19_38_1, de19_38_2, de19_38_3]

    # layer3 = [conv13, nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
    # layer4 = [conv14, nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
    layers = [layer1, layer21, layer22, layer23, layer24, layer25, layer26, layer27, layer3, layer4, layer5, layer6]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    cc0 = torch.nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
    cc0_1 = nn.BatchNorm2d(256)
    cc0_2 = nn.ReLU(inplace=True)

    cc1 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    cc1_0 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    cc1_1 = nn.BatchNorm2d(512)
    cc1_2 = nn.ReLU(inplace=True)

    cc2 = torch.nn.Conv2d(512, 192, kernel_size=(1, 1), stride=(1, 1))
    cc2_1 = nn.BatchNorm2d(256)
    cc2_2 = nn.ReLU(inplace=True)

    cc3 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    cc3_0 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    cc3_1 = nn.BatchNorm2d(512)
    cc3_2 = nn.ReLU(inplace=True)

    cc4 = torch.nn.Conv2d(512, 192, kernel_size=(1, 1), stride=(1, 1))
    cc4_1 = nn.BatchNorm2d(256)
    cc4_2 = nn.ReLU(inplace=True)

    cc5 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
    cc5_0 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    cc5_1 = nn.BatchNorm2d(512)
    cc5_2 = nn.ReLU(inplace=True)
    '''
    cc6 = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    cc6_1 = nn.BatchNorm2d(256)
    cc6_2 = nn.ReLU(inplace=True)
    cc7 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
    cc7_0 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    cc7_1 = nn.BatchNorm2d(512)
    cc7_2 = nn.ReLU(inplace=True)
    '''
    cc8 = torch.nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
    cc8_1 = nn.BatchNorm2d(1024)
    cc8_2 = torch.nn.Conv2d(1280, 1024, kernel_size=(1, 1), stride=(1, 1))
    cc8_3 = nn.BatchNorm2d(1024)

    cc9 = torch.nn.Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1))
    cc9_1 = nn.BatchNorm2d(512)

    cc10 = torch.nn.Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1))
    cc10_1 = nn.BatchNorm2d(512)

    cc11 = torch.nn.Conv2d(640, 512, kernel_size=(1, 1), stride=(1, 1))
    cc11_1 = nn.BatchNorm2d(512)
    '''
    cc12 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    cc12_1 = nn.BatchNorm2d(512)
    '''

    layers = [cc0, cc0_1, cc0_2,
              cc1, cc1_0, cc1_1, cc1_2,
              cc2, cc2_1, cc2_2,
              cc3, cc3_0, cc3_1, cc3_2,
              cc4, cc4_1, cc4_2,
              cc5, cc5_0, cc5_1, cc5_2,

              cc8, cc8_1, cc8_2, cc8_3, nn.ReLU(inplace=True), cc9, cc9_1, nn.ReLU(inplace=True), cc10, cc10_1,
              nn.ReLU(inplace=True), cc11, cc11_1, nn.ReLU(inplace=True)
              ]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = [
        torch.nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        # torch.nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        # torch.nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        # ,
        torch.nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        # torch.nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    ]
    conf_layers = [
        torch.nn.Conv2d(512, 6*21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        # torch.nn.Conv2d(512, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        # torch.nn.Conv2d(512, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.Conv2d(1024, 6*21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.Conv2d(512, 6*21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.Conv2d(512, 6*21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        # ,
        torch.nn.Conv2d(512, 6*21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        # torch.nn.Conv2d(512, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    ]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '320':[],
    '300': [],
    '512': [],
}
extras = {
    '320': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '320': [4, 6, 6, 6, 4, 4],
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    #if size != 300:
    #print("ERROR: You specified size " + repr(size) + ". However, " +
     #         "currently only SSD300 (size=300) is supported!")
      #  return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes,resnet18())
