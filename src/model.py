import torch.nn as nn
import torch
import torch.nn.functional as F
from mobilev4 import *
from tinyUnet import UNetDecoder


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = mobilenetv4_medium()

    def forward(self, inputs):
        return self.cnn(inputs)


class Attention(nn.Module):
    def __init__(self, nc, K=8):
        super(Attention, self).__init__()
        self.K = K

        channels = [256, 128]
        self.conv_pool_1 = nn.Sequential(
            nn.Conv2d(nc, channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_pool_2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 1, 1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
        )

        self.cnn_1_1 = nn.Conv2d(channels[0], 64, 1, 1, 0)

        self.deconv1 = nn.ConvTranspose2d(channels[1], 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, self.K, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(self.K)

        self.relu    = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.LazyLinear(48)
        self.fc2 = nn.LazyLinear(48)

    def forward(self, inputs):
        x0 = self.conv_pool_1(inputs)
        x1 = self.conv_pool_2(x0)

        N, C, H, W = x1.shape
        fc_x = x1.view(N, C, H*W)

        # fc1 = nn.LazyLinear(H*W)
        # fc2 = nn.LazyLinear(H*W)
        fc_atten = self.fc2((self.fc1(fc_x))).reshape(N, C, H, W)

        score = self.relu(self.deconv1(fc_atten))
        score = self.bn1(score + self.cnn_1_1(x0))
        atten = self.softmax(self.bn2(self.deconv2(score)))
        return atten


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
    def __init__(self, nclass, K=8):
        super(LPR_model, self).__init__()
        self.K = K
        self.encoder = Encoder()
        self.attention = Attention(nc=256, K=K)
        self.cls_decoder = ClsDecoder(nclass)
        self.seg_decoder = SegDecoder(K)

    def forward(self, inputs):
        e1, _, e3, e4, e5 = self.encoder(inputs)

        atten = self.attention(e5)
        # atten_list = torch.chunk(atten, self.K, 1)

        preds = self.cls_decoder(atten, e5)
        seg = self.seg_decoder(e5, e4, e3, e1)
        return preds, seg


if __name__ == "__main__":
    model = LPR_model(68, 8)
    inputs = torch.randn(1, 1, 32, 96)
    inputs = torch.randn(1, 1, 128, 384)
    preds, seg = model(inputs)

    print(preds.shape, seg.shape)
    # export as onnx
    torch.onnx.export(model, inputs, "lpr.onnx", verbose=False, opset_version=20)
