import torch.nn as nn
import torch
import torch.nn.functional as F
from mobilev4 import *


# conv2d bn relu
def ConvBN(in_channels, out_channels, kernel_size, stride, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Encoder(nn.Module):
    def __init__(self, nc=1):
        super(Encoder, self).__init__()
        self.cnn = mobilenetv4_medium()

    def forward(self, input):
        conv_out = self.cnn(input)
        return conv_out


class Attention(nn.Module):
    def __init__(self, nc, K=8, downsample=4):
        super(Attention, self).__init__()
        self.K = K

        # atten_0.add_module('conv_a_0',nn.Conv2d(nc, nm[1], 3, 1, 1))
        # atten_0.add_module('bn_a_0', nn.BatchNorm2d(nm[1]))
        # atten_0.add_module('relu_a_0', nn.ReLU(True))
        # atten_0.add_module('pooling_a_0',nn.MaxPool2d((2, 2)))

        # atten_1.add_module('conv_a_1',nn.Conv2d(nm[1], nm[2], 3, 1, 1))
        # atten_1.add_module('bn_a_1', nn.BatchNorm2d(nm[2]))
        # atten_1.add_module('relu_a_1', nn.ReLU(True))
        # atten_1.add_module('pooling_a_1',nn.MaxPool2d((2, 2)))

        nm = [256,256,160]
        self.atten_0 = nn.Sequential(
            ConvBN(nc, nm[0], 3, 1, 1),
            nn.MaxPool2d((2, 2))
        )
        self.atten_1 = nn.Sequential(
            ConvBN(nm[0], nm[1], 3, 1, 1),
            nn.MaxPool2d((2, 2))
        )

        # fc_dim = int(96*32/downsample/downsample/16)
        # self.atten_fc1 = nn.Linear(fc_dim, fc_dim)
        # self.atten_fc2 = nn.Linear(fc_dim, fc_dim)

        self.cnn_1_1 = nn.Conv2d(nm[1],64,1,1,0)

        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.deconv1 = nn.ConvTranspose2d(nm[2], 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
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
        atten = self.sigmoid(self.bn2(self.deconv2(score)))

        return atten


class Decoder(nn.Module):
    def __init__(self, nclass, input_dim=256, K=8):
        super(Decoder, self).__init__()
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


class LPR_model(nn.Module):
    def __init__(self, nc, nclass, imgH=32, imgW=96, K=8):
        super(LPR_model, self).__init__()
        self.K = K
        self.encoder = Encoder(nc)
        self.attention = Attention(nc=256, K=K, downsample=4)
        self.decoder = Decoder(nclass, input_dim=256)

    def forward(self, x):
        feats = self.encoder(x)
        print("feats.size", feats.size())
        atten = self.attention(feats)
        atten_list = torch.chunk(atten, self.K, 1)
        preds = self.decoder(atten, feats)
        return preds, atten_list


if __name__ == "__main__":
    model = LPR_model(3, 68, 128, 384, 8)
    inputs = torch.randn(1, 3, 128, 384)
    preds = model(inputs)

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
