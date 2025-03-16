import torch.nn as nn
import torch
import torch.nn.functional as F
from mobilev4 import *
from tinyUnet import UNetDecoder


class Attention(nn.Module):
    def __init__(self, channels, temporal, **kwargs):
        super(Attention, self).__init__()
        self.temporal = temporal
        self.cba = nn.Sequential(
            nn.LazyConv2d(channels, 3, 2, 1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(inplace=True),
        )
        self.ll = nn.Sequential(
            nn.LazyLinear(12),
            nn.LazyLinear(12),
        )
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.LazyConv2d(channels, 1, 1, 0),
            nn.ReLU(inplace=True),
        )
        self.bn = nn.BatchNorm2d(channels)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.LazyConv2d(temporal, 1, 1, 0),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        x = self.cba(inputs)
        bs, c, h, w = x.size()

        x = x.view(bs, c, -1)
        x = self.ll(x)
        x = x.view(bs, c, h, w)

        x = self.up0(x) + inputs
        x = self.up1(self.bn(x))
        return x


class ClsDecoder(nn.Module):
    def __init__(self, nclass, K=8):
        super(ClsDecoder, self).__init__()
        self.K = K
        self.fc = nn.LazyLinear(nclass)

    def forward(self, atten, conv_out):
        N = conv_out.size(0)
        atten = atten.reshape(N, self.K, -1)

        conv_out = conv_out.reshape(conv_out.size(0), conv_out.size(1), -1)
        conv_out = conv_out.permute(0,2,1)

        atten_out = torch.matmul(atten, conv_out)
        atten_out = atten_out.view(N, self.K, -1)
        return self.fc(atten_out)


class SegDecoder(nn.Module):
    def __init__(self, num_classes=10, in_filters=[160, 336, 416], out_filters=[64, 128, 256]):
        super(SegDecoder, self).__init__()
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0])
        self.last_conv = nn.LazyConv2d(num_classes, kernel_size=1)

    def forward(self, x, skip3, skip2, skip1):
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.last_conv(x)
        return x


class LPR_model(nn.Module):
    def __init__(self, nclass, seq_len=8):
        super(LPR_model, self).__init__()
        self.encoder = mobilenetv4_medium()
        self.attention = Attention(channels=256, temporal=seq_len)
        self.cls_decoder = ClsDecoder(nclass)
        self.seg_decoder = SegDecoder(seq_len)

    def forward(self, inputs):
        e1, _, e3, e4, e5 = self.encoder(inputs)
        print("e5 size: ", e5.size())

        atten = self.attention(e5)

        cls_head = self.cls_decoder(atten, e4)
        seg_head = self.seg_decoder(e5, e4, e3, e1)
        return cls_head, seg_head


if __name__ == "__main__":
    model = LPR_model(68, 8)
    input_shape = (1, 1, 32, 96)
    inputs = torch.randn(input_shape)
    cls_head, seg_head = model(inputs)

    import sys
    sys.path.append("D:/projects/SALPR/utils")
    print(sys.path)
    from tools import count_parameters

    count_parameters(model, input_size=input_shape)

    print("cls_head size: ", cls_head.size())
    print("seg_head size: ", seg_head.size())
    # export as onnx
    torch.onnx.export(model, inputs, "lpr.onnx", verbose=False, opset_version=20)
