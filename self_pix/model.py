import torch.nn as nn
import torch
import torch.nn.functional as F
from mobilev4 import *


# def dot_product(seg, cls):
#     b, n, h, w = seg.shape
#     seg = seg.view(b, n, -1)
#     cls = cls.unsqueeze(-1)  # Add an extra dimension for broadcasting
#     final = torch.einsum("bik,bi->bik", seg, cls)
#     final = final.view(b, n, h, w)
#     return final


# conv2d bn relu
def ConvBN(in_channels, out_channels, kernel_size, stride, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def LazyConvBN(out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same'):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, nc=1):
        super(Encoder, self).__init__()
        # self.cnn = mobilenetv4_small()
        self.cnn = mobilenetv4_medium()
        # self.cnn = mobilenetv4_large()

    def forward(self, input):
        conv_out = self.cnn(input)
        return conv_out


class Attention(nn.Module):
    def __init__(self, nc, K=8, downsample=4):
        super(Attention, self).__init__()
        self.K = K

        # nm = [128, 64]
        nm = [256, 128]
        # nm = [512, 192]
        self.atten_0 = nn.Sequential(
            ConvBN(nc, nm[0], 3, 1, 1),
            nn.MaxPool2d((2, 2))
        )
        self.atten_1 = nn.Sequential(
            ConvBN(nm[0], nm[1], 3, 1, 1),
            nn.MaxPool2d((2, 2))
        )

        self.cnn_1_1 = nn.Conv2d(nm[0],64,1,1,0)

        self.relu    = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.deconv1 = nn.ConvTranspose2d(nm[1], 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, self.K, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(self.K)

    def forward(self, x):
        batch_size = x.size(0)

        x0 = self.atten_0(x)
        x1 = self.atten_1(x0)

        channel, height, width = x1.size(1), x1.size(2), x1.size(3)
        fc_x = x1.view(batch_size, channel, -1)

        fc_dim = fc_x.size(2)
        atten_fc1 = nn.Linear(fc_dim, fc_dim)
        atten_fc2 = nn.Linear(fc_dim, fc_dim)
        fc_atten = atten_fc2((atten_fc1(fc_x)))

        fc_atten = fc_atten.reshape(batch_size, channel, height, width)

        score = self.relu(self.deconv1(fc_atten))
        score = self.bn1(score + self.cnn_1_1(x0))
        atten = self.softmax(self.bn2(self.deconv2(score)))
        return atten


class DecoderText(nn.Module):
    def __init__(self, nclass, input_dim=256, K=8):
        super(DecoderText, self).__init__()
        self.input_dim = input_dim
        self.nclass = nclass
        self.K = K
        self.fc = nn.Linear(self.input_dim, self.nclass)

    def forward(self, atten, conv_out):
        batch_size = conv_out.size(0)
        atten = atten.reshape(batch_size, self.K, -1)

        conv_out = conv_out.reshape(conv_out.size(0), conv_out.size(1), -1)
        conv_out = conv_out.permute(0,2,1)

        atten_out = torch.bmm(atten, conv_out)
        atten_out = atten_out.view(batch_size, self.K, -1)
        return self.fc(atten_out)


class DecoderSeg(nn.Module):
    def __init__(self, cat_channels=32, input_dim=256, K=8):
        super(DecoderSeg, self).__init__()
        self.input_dim = input_dim
        self.cat_channels = 32
        self.K = K
        # unet like decoder
        self.d4 = nn.ModuleList([
            LazyConvBN(self.cat_channels),
            LazyConvBN(self.cat_channels),
            LazyConvBN(self.cat_channels),
        ])
        self.d4_conv = LazyConvBN(32 * len(self.d4))
        self.d3 = nn.ModuleList([
            LazyConvBN(self.cat_channels),
            LazyConvBN(self.cat_channels),
            LazyConvBN(self.cat_channels),
        ])
        self.d3_conv = LazyConvBN(32 * len(self.d3))
        self.d2 = nn.ModuleList([
            LazyConvBN(self.cat_channels),
            LazyConvBN(self.cat_channels),
            LazyConvBN(self.cat_channels),
        ])
        self.d2_conv = LazyConvBN(32 * len(self.d2))
        self.d1 = nn.ModuleList([
            LazyConvBN(self.cat_channels),
            LazyConvBN(self.cat_channels),
            LazyConvBN(self.cat_channels),
            # LazyConvBN(self.cat_channels),
        ])
        self.d1_conv = LazyConvBN(32 * len(self.d1))
        # self.last_conv = nn.LazyConv2d(2, kernel_size=1)

        # Classification Guided Module
        self.cgm = nn.Sequential(
            nn.Dropout(0.5),
            nn.LazyConv2d(2, kernel_size=1, padding=0),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, e1, e2, e3, e4, e5, atten):
        d4 = [
            F.max_pool2d(e3, 2),
            e4,
            F.interpolate(e5, scale_factor=2, mode='bilinear', align_corners=True)
        ]
        d4 = [conv(d) for conv, d in zip(self.d4, d4)]
        d4 = self.d4_conv(torch.cat(d4, dim=1))

        d3 = [
            F.max_pool2d(e2, 2),
            e3,
            F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        ]
        d3 = [conv(d) for conv, d in zip(self.d3, d3)]
        d3 = self.d3_conv(torch.cat(d3, dim=1))

        d2 = [
            F.max_pool2d(e1, 2),
            e2,
            F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        ]
        d2 = [conv(d) for conv, d in zip(self.d2, d2)]
        d2 = self.d2_conv(torch.cat(d2, dim=1))

        d1 = [
            e1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
        ]
        d1 = [conv(d) for conv, d in zip(self.d1, d1)]
        d1 = self.d1_conv(torch.cat(d1, dim=1))
        # d1 = self.last_conv(d1)

        up_d1 = nn.Sequential(
            nn.LazyConv2d(2, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )(d1)
        up_d2 = nn.Sequential(
            nn.LazyConv2d(2, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Sigmoid(),
        )(d2)
        up_d3 = nn.Sequential(
            nn.LazyConv2d(2, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Sigmoid(),
        )(d3)
        up_d4 = nn.Sequential(
            nn.LazyConv2d(2, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Sigmoid(),
        )(d4)
        up_d5 = nn.Sequential(
            nn.LazyConv2d(2, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
            nn.Sigmoid(),
        )(e5)

        # cat all
        cat = torch.cat([up_d1, up_d2, up_d3, up_d4, up_d5], dim=1)
        return cat


class LPR_model(nn.Module):
    def __init__(self, nc, nclass, imgH=32, imgW=96, K=8):
        super(LPR_model, self).__init__()
        self.K = K
        self.encoder = Encoder(nc)
        self.attention = Attention(nc=256, K=K, downsample=4)
        self.text_decoder = DecoderText(nclass, input_dim=256)
        self.mask_decoder = DecoderSeg(input_dim=256, K=K)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        print(e1.shape, e2.shape, e3.shape, e4.shape, e5.shape)
        atten = self.attention(e5)
        p_text = self.text_decoder(atten, e5)
        p_mask = self.mask_decoder(e1, e2, e3, e4, e5, atten)
        print("atten: ", atten.shape)
        print("argmax: ", torch.argmax(atten, dim=1).shape)
        # return preds, torch.chunk(atten, self.K, 1)
        # return p_text, p_mask
        return p_text


if __name__ == "__main__":
    model = LPR_model(3, 68, 128, 384, 8)
    inputs_shape = (1, 3, 128, 384)
    inputs = torch.randn(inputs_shape)
    outputs = model(inputs)

    quit()

    # if isinstance(outputs, tuple):
    #     for i, output in enumerate(outputs):
    #         print(f"Stage {i + 1} output size: {output.size()}")
    # else:
    #     print(outputs.size())

    import sys
    sys.path.append('utils')
    from tools import *

    count_parameters(model, inputs_shape)

    # import os, glob, random
    # import numpy as np
    # from PIL import Image
    # import matplotlib.pyplot as plt

    # img_path = random.choice(glob.glob('images/*.jpg'))
    # img = Image.open(img_path).resize((96, 32)).convert('L')
    # img = np.array(img).reshape(1, 1, 32, 96).astype(np.float32) / 255.0

    # model.load_state_dict(torch.load('self_pix/model_85_acc_0.9954.pth', weights_only=True, map_location='cpu'), strict=False)
    # preds, atten_list = model(torch.from_numpy(img))
    # preds = torch.argmax(preds, dim=2).squeeze().detach().numpy()

    # fig, axs = plt.subplots(1, 9, figsize=(16, 2))
    # axs[0].imshow(img.squeeze(), cmap='gray')
    # axs[0].axis('off')

    # for i, atten in enumerate(atten_list):
    #     idx = i + 1
    #     atten = atten.squeeze().detach().numpy()
    #     atten = (atten - atten.min()) / (atten.max() - atten.min())
    #     atten = (atten * 255).astype(np.uint8)
    #     img = Image.fromarray(atten).resize((96, 32), Image.NEAREST)
    #     axs[idx].set_title(preds[i])
    #     axs[idx].imshow(img, cmap='gray')
    #     axs[idx].axis('off')

    # plt.show()
    # print(img_path, preds)
