import torch.nn as nn
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, nc=1):
        super(Encoder, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1, 1, 1, 1] 
        ss = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        nm = [32, 32, 32, 64, 64, 64, 128, 128, 128]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        convRelu(1)
        convRelu(2)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 32*100*20
        convRelu(3)
        convRelu(4)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 64*50*10
        convRelu(6)
        convRelu(7)
        convRelu(8)

        self.cnn = cnn

    def forward(self, input):
        conv_out = self.cnn(input)
        return conv_out


class Attention(nn.Module):
    def __init__(self, nc, K=8, downsample=4):
        super(Attention, self).__init__()
        self.K = K

        nm = [256,128]
        atten_0 = nn.Sequential()
        atten_0.add_module('conv_a_0',nn.Conv2d(nc, nm[0], 3, 1, 1))
        atten_0.add_module('bn_a_0', nn.BatchNorm2d(nm[0]))
        atten_0.add_module('relu_a_0', nn.ReLU(True))
        atten_0.add_module('pooling_a_0',nn.MaxPool2d((2, 2)))

        atten_1 = nn.Sequential()
        atten_1.add_module('conv_a_1',nn.Conv2d(nm[0], nm[1], 3, 1, 1))
        atten_1.add_module('bn_a_1', nn.BatchNorm2d(nm[1]))
        atten_1.add_module('relu_a_1', nn.ReLU(True))
        atten_1.add_module('pooling_a_1',nn.MaxPool2d((2, 2)))

        self.atten_0 = atten_0
        self.atten_1 = atten_1

        fc_dim = int(96*32/downsample/downsample/16)
        self.atten_fc1 = nn.Linear(fc_dim, fc_dim)
        self.atten_fc2 = nn.Linear(fc_dim, fc_dim)

        self.cnn_1_1 = nn.Conv2d(nm[0],64,1,1,0)

        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.deconv1 = nn.ConvTranspose2d(nm[1], 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, self.K, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn2     = nn.BatchNorm2d(self.K)

    def forward(self, inputs):
        conv_out = inputs

        x0 = self.atten_0(inputs)
        x1 = self.atten_1(x0)

        N, C, H, W = x1.shape
        fc_x = x1.view(N, C, -1)

        fc_atten = self.atten_fc2(self.atten_fc1(fc_x))
        fc_atten = fc_atten.reshape(N, C, H, W)

        score = self.relu(self.deconv1(fc_atten))
        score = self.bn1(score + self.cnn_1_1(x0))
        atten = self.sigmoid(self.deconv2(score))

        atten_list = torch.chunk(atten, self.K, 1)
        atten = atten.reshape(N, self.K, -1)

        inputs = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        inputs = inputs.permute(0,2,1)

        atten = torch.matmul(atten, inputs)
        atten = atten.view(N, self.K, -1)
        return atten_list, atten


class Decoder(nn.Module):
    def __init__(self, nclass, input_dim=512, K =8):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.nclass = nclass
        self.fc = nn.Linear(self.input_dim, self.nclass)

    def forward(self, inputs):
        return self.fc(inputs)


class LPR_model(nn.Module):
    def __init__(self, nc, nclass, K=8):
        super(LPR_model, self).__init__()

        self.encoder = Encoder(nc)
        self.attention = Attention(nc=128, K=K, downsample=4)
        self.decoder = Decoder(nclass, input_dim=128)

    def forward(self, input):
        conv_out = self.encoder(input)
        atten_list, atten_out = self.attention(conv_out)
        preds = self.decoder(atten_out)
        return preds, atten_list


if __name__ == "__main__":
    model = LPR_model(1, 68, 8).eval()
    inputs = torch.randn(1, 1, 32, 96)
    # preds = model(inputs)

    import glob, random
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # Load a random image
    img_path = random.choice(glob.glob('data/*.jpg'))
    img = np.array(Image.open(img_path).resize((96, 32)).convert('L')).astype(np.float32) / 255.0
    img = torch.from_numpy(img.reshape(1, 1, 32, 96))

    # Load model weights and get predictions
    model.load_state_dict(torch.load('weights/model_85_acc_0.9954.pth', map_location='cpu'), strict=False)
    preds, attens = model(img)
    preds = torch.argmax(preds, dim=2).squeeze().detach().numpy()

    # Prepare figure
    fig, axs = plt.subplots(1, 9, figsize=(16, 2))
    axs[0].imshow(img.squeeze(), cmap='gray')
    axs[0].axis('off')

    # Display attention maps
    for i, atten in enumerate(attens):
        idx = i + 1
        atten = atten.squeeze().detach().numpy()
        atten = (atten - atten.min()) / (atten.max() - atten.min())
        img = Image.fromarray((atten * 255).astype(np.uint8)).resize((96, 32), Image.NEAREST)
        axs[idx].set_title(preds[i])
        axs[idx].imshow(img, cmap='gray')
        axs[idx].axis('off')

    plt.tight_layout()
    plt.savefig('result.png')
    print(img_path, preds)
