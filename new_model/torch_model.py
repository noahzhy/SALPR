import torch
import torch.nn as nn
import math


__all__ = ['mobilenetv4_small']


def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class UniversalInvertedBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 stride,
                 middle_dw_downsample: bool = True,
                 use_layer_scale: bool = False,
                 layer_scale_init_value: float = 1e-5):
        super(UniversalInvertedBottleneck, self).__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size

        if start_dw_kernel_size:
           self.start_dw_conv = nn.Conv2d(in_channels, in_channels, start_dw_kernel_size, 
                                          stride if not middle_dw_downsample else 1,
                                          (start_dw_kernel_size - 1) // 2,
                                          groups=in_channels, bias=False)
           self.start_dw_norm = nn.BatchNorm2d(in_channels)
        
        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = nn.Conv2d(in_channels, expand_channels, 1, 1, bias=False)
        self.expand_norm = nn.BatchNorm2d(expand_channels)
        self.expand_act = nn.ReLU(inplace=True)

        if middle_dw_kernel_size:
           self.middle_dw_conv = nn.Conv2d(expand_channels, expand_channels, middle_dw_kernel_size,
                                           stride if middle_dw_downsample else 1,
                                           (middle_dw_kernel_size - 1) // 2,
                                           groups=expand_channels, bias=False)
           self.middle_dw_norm = nn.BatchNorm2d(expand_channels)
           self.middle_dw_act = nn.ReLU(inplace=True)
        
        self.proj_conv = nn.Conv2d(expand_channels, out_channels, 1, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(out_channels)

        if use_layer_scale:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

        self.use_layer_scale = use_layer_scale
        self.identity = stride == 1 and in_channels == out_channels

    def forward(self, x):
        shortcut = x

        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_norm(x)

        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_norm(x)
            x = self.middle_dw_act(x)

        x = self.proj_conv(x)
        x = self.proj_norm(x)

        if self.use_layer_scale:
            x = self.gamma * x

        return x + shortcut if self.identity else x


class MobileNetV4(nn.Module):
    def __init__(self, cin=1,cout=128):
        super(MobileNetV4, self).__init__()

        block_specs = [
            # conv_bn, kernel_size, stride, out_channels
            # uib, start_ks, middle_ks, stride, out_channels, expand_ratio
            # stage 1
            ('conv_bn', 3, 1, 24),
            ('conv_bn', 3, 2, 64),
            ('conv_bn', 1, 1, 32),
            #
            ('uib', 5, 5, 2, 64, 3.0),  # ExtraDW
            ('uib', 0, 3, 1, 64, 2.0),  # IB
            ('uib', 0, 3, 1, 64, 2.0),  # IB
            ('uib', 3, 0, 1, 64, 4.0),  # ConvNext
            #
            ('uib', 3, 3, 2, 96, 6.0),  # ExtraDW
            ('uib', 0, 5, 1, 96, 3.0),  # IB
            ('conv_bn', 1, 1, cout),  # Conv
        ]

        self.stage1 = nn.Sequential(
            ConvBN(cin, 24, 3, 1),
            ConvBN(24, 64, 3, 2),
            ConvBN(64, 32, 1, 1),
        )

        self.stage2 = nn.Sequential(
            UniversalInvertedBottleneck(32, 64, 3.0, 5, 5, 2),
            UniversalInvertedBottleneck(64, 64, 2.0, 0, 3, 1),
            UniversalInvertedBottleneck(64, 64, 2.0, 0, 3, 1),
            UniversalInvertedBottleneck(64, 64, 4.0, 3, 0, 1),
        )

        self.stage3 = nn.Sequential(
            UniversalInvertedBottleneck(64, 96, 6.0, 3, 3, 2),
            UniversalInvertedBottleneck(96, 96, 3.0, 0, 5, 1),
            ConvBN(96, cout, 1, 1),
        )

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s1, s2, s3


class Attention(nn.Module):
    def __init__(self, channels, temporal, **kwargs):
        super(Attention, self).__init__()
        self.temporal = temporal
        self.cba = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.ll = nn.Sequential(
            nn.Linear(12, 12),
            nn.Linear(12, 12),
        )
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels, channels, 1, 1, 0),
        )
        self.up1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels, temporal, 1, 1, 0),
            nn.Softmax(dim=1)
            # nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.cba(inputs)
        bs, c, h, w = x.size()

        x = x.view(bs, c, -1)
        x = self.ll(x)
        x = x.view(bs, c, h, w)

        x = self.up0(x) + inputs
        x = self.up1(x)
        return x


class TinyLPR(nn.Module):
    def __init__(self, T=8, n_class=68, n_feat=128):
        super(TinyLPR, self).__init__()
        self.T = T
        self.n_class = n_class
        self.n_feat = n_feat

        self.backbone = MobileNetV4(1, n_feat)
        self.attention = Attention(n_feat, T)
        self.out = nn.Linear(n_feat, n_class)

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(64, n_feat, 1, 1, 0),
        # )

    def forward(self, x):
        _, s2, s3 = self.backbone(x)
        bs, c, h, w = s3.size()

        attn = self.attention(s3)
        attn = attn.reshape(bs, self.T, -1)

        # shortcut = self.conv1(s2)
        shortcut = s3.reshape(bs, c, -1).permute(0, 2, 1)
        attn_out = torch.bmm(attn, shortcut).view(bs, self.T, -1)
        return self.out(attn_out)


if __name__ == '__main__':
    model = TinyLPR()
    print(model)
    x = torch.randn(1, 1, 32, 96)
    y = model(x)
    print(y.size())

    # quit()

    # export model as onnx
    import torch.onnx

    torch.onnx.export(model, x,
        "mobilenetv4_small.onnx",
        verbose=False,
        input_names=["input"],
        output_names=["output"]
    )

    # count number of parameters and flops
    from thop import profile
    macs, params = profile(model, inputs=(x,))
    print('MACs: {} MFLOPs'.format(macs / 1e6))
    print('Params: {} M'.format(params / 1e6))

