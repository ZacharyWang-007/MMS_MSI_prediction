import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MixOrderChannel(nn.Module):
    def __init__(self, channel):
        super(MixOrderChannel, self).__init__()
        self.order1 = SELayer(channel, 4)
        self.order2 = SELayer(channel, 8)
        self.order3 = SELayer(channel, 16)

    def forward(self, x):
        scale = (self.order1(x) + self.order2(x) + self.order3(x))/3
        return scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) * x
        return scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class Extractor1(nn.Module):
    def __init__(self):
        super(Extractor1, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)

        self.pre_layer_res = nn.Sequential(*list(resnet.children())[0:4])
        self.res_block1 = nn.Sequential(*list(resnet.children())[4])
        self.res_block2 = nn.Sequential(*list(resnet.children())[5])
        self.res_block3 = nn.Sequential(*list(resnet.children())[6])
        self.res_block4 = nn.Sequential(*list(resnet.children())[7])

        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))

        self.spatial1 = SpatialGate()
        self.spatial2 = SpatialGate()
        self.spatial3 = SpatialGate()
        self.spatial4 = SpatialGate()

        self.channel1 = MixOrderChannel(channel=64)
        self.channel2 = MixOrderChannel(channel=128)
        self.channel3 = MixOrderChannel(channel=256)
        self.channel4 = MixOrderChannel(channel=512)

        # self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.pre_layer_res(x)
        # x = self.spatial0(x)
        x = self.res_block1(x)
        x = (self.spatial1(x) + self.channel1(x)) + x
        x = self.res_block2(x)
        x = (self.spatial2(x) + self.channel2(x)) + x
        x = self.res_block3(x)
        x = (self.spatial3(x) + self.channel3(x)) + x
        x = self.res_block4(x)
        x = (self.spatial4(x) + self.channel4(x)) + x

        return x


class Model(nn.Module):
    def __init__(self, size_arg="big", dropout=0.25, n_classes=1):
        super(Model, self).__init__()

        self.extractor_tumour = Extractor1()

        self.compress0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.25),
        )

        self.linear1 = nn.Sequential(nn.Linear(47, 64),
                                     # nn.LayerNorm(64),
                                     nn.BatchNorm1d(64),
                                     # nn.ReLU(inplace=True),
                                     nn.LeakyReLU(0.25),
                                     nn.Dropout(0.25)
                                     )

        self.linear2 = nn.Sequential(nn.Linear(256 + 64, 256),
                                     nn.LayerNorm(256),
                                     # nn.BatchNorm1d(256),
                                     # nn.ReLU(inplace=True),
                                     nn.LeakyReLU(0.25),
                                     nn.Dropout(0.25)
                                     )

        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpooling = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Linear(256, 1, bias=False)
        # self.classifier = nn.Linear(256, 1, bias=True)

        self.channel_attention = MixOrderChannel(channel=256 + 64)

    def forward(self, data, return_hms=False):
        feats_t, clinical_data = data

        feats_tumour = self.extractor_tumour(feats_t)  # feats_n_tumour
        if return_hms:
            return feats_tumour

        feats_tumour = self.compress0(feats_tumour)
        feats_tumour = self.avgpooling(feats_tumour).squeeze(-1).squeeze(-1)

        clinical_data = self.linear1(clinical_data)

        feats = torch.cat((clinical_data, feats_tumour), dim=1)

        feats = self.channel_attention(feats[:, :, None, None])

        feats = self.linear2(feats.squeeze())

        logits = self.classifier(feats)

        return logits


if __name__ == '__main__':
    mil = Model()
    input = torch.randn(100, 1024)
    logits = mil(input)

    loss = nn.CrossEntropyLoss()
    target = torch.empty(1, dtype=torch.long).random_(2)
    output = loss(logits, target)

