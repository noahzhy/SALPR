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
            nn.LazyBatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.LazyLinear(12),
            nn.ReLU6(inplace=True),
            nn.LazyLinear(12),
            nn.Hardtanh(min_val=0, max_val=1),
        )
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.LazyConv2d(channels, 1, 1, 0),
            nn.LazyBatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.LazyConv2d(temporal, 1, 1, 0),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        x = self.cba(inputs)
        bs, c, h, w = x.size()

        x = x.view(bs, c, -1)
        x = self.mlp(x)
        x = x.view(bs, c, h, w)

        x = self.up0(x) + inputs
        x = self.up1(x)
        return x


class ClsDecoder(nn.Module):
    def __init__(self, nclass, T=8):
        super(ClsDecoder, self).__init__()
        self.T = T
        self.fc = nn.LazyLinear(nclass)

    def forward(self, atten, shortcut):
        N, C, H, W = shortcut.size()
        atten = atten.reshape(N, self.T, -1)

        shortcut = shortcut.reshape(N, C, -1)
        shortcut = shortcut.permute(0, 2, 1)

        atten = torch.matmul(atten, shortcut)
        atten = atten.view(N, self.T, -1)
        return self.fc(atten)


class SegDecoder(nn.Module):
    def __init__(self,
        num_classes=10,
        in_filters=[416, 208, 96],
        out_filters=[128, 64, 32, 24, 16],
    ):
        super(SegDecoder, self).__init__()
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0])
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder4 = nn.Sequential(
            nn.LazyConvTranspose2d(out_filters[3], 2, 2, 0),
            nn.LazyBatchNorm2d(out_filters[3]),
            nn.ReLU(inplace=True),
        )
        self.decoder5 = nn.Sequential(
            nn.LazyConvTranspose2d(out_filters[4], 2, 2, 0),
            nn.LazyBatchNorm2d(out_filters[4]),
            nn.ReLU(inplace=True),
        )
        self.last_conv = nn.LazyConv2d(num_classes, kernel_size=1)

    def forward(self, x, skip3, skip2, skip1):
        x = self.decoder1(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder3(x, skip1)
        x = self.decoder4(x)
        x = self.decoder5(x)
        x = self.last_conv(x)
        return x


class LPR_model(nn.Module):
    def __init__(self, nclass, seq_len=8):
        super(LPR_model, self).__init__()
        self.encoder    = mobilenetv4_medium()
        self.attention  = Attention(channels=256, temporal=seq_len)
        self.cls_decoder= ClsDecoder(nclass, seq_len)
        self.seg_decoder= SegDecoder(seq_len)

    def forward(self, inputs):
        e1, _, e3, e4, e5 = self.encoder(inputs)

        atten = self.attention(e5)
        print("atten size: ", atten.size())

        cls_head = self.cls_decoder(atten, e4)
        seg_head = self.seg_decoder(e5, e4, e3, e1)
        # TODO: seg_head
        return cls_head, seg_head


if __name__ == "__main__":
    model = LPR_model(nclass=68, seq_len=8)
    input_shape = (1, 1, 32, 96)
    inputs = torch.randn(input_shape)
    cls_head, seg_head = model(inputs)
    print("cls_head size: ", cls_head.size())
    print("seg_head size: ", seg_head.size())

    import sys
    sys.path.append("D:/projects/SALPR/utils")
    from tools import count_parameters

    count_parameters(model, input_size=input_shape)
    # export as onnx
    torch.onnx.export(model, inputs, "lpr.onnx", verbose=False, opset_version=20)
