import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .Module import *

class ConvNeXtFreqBranch(nn.Module):
    def __init__(self, backbone_name="convnext_tiny.in12k_ft_in1k_384", in_channels=12):
        super().__init__()

        full_model = timm.create_model(backbone_name, pretrained=True, features_only=False)

        # 修改 stem 卷积为支持 12 通道输入
        if hasattr(full_model, "stem"):
            old_conv = full_model.stem[0]
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=2,
                padding=2,
                bias=old_conv.bias is not None
            )
            full_model.stem[0] = new_conv
        else:
            raise ValueError("ConvNeXt 模型中不包含 stem 属性")

        self.stem = full_model.stem
        self.stages = full_model.stages  # 对应四个阶段的特征提取器

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs  # 返回 [f1, f2, f3, f4]

class DSWFNet(nn.Module):
    def __init__(self, backbone_name="convnext_tiny.in12k_ft_in1k_384", num_classes=1):
        super().__init__()

        self.spatial_backbone = timm.create_model(
            model_name=backbone_name,
            features_only=True,
            pretrained=True,
            out_indices=(0, 1, 2, 3)
        )

        self.freq_backbone = ConvNeXtFreqBranch(backbone_name="convnext_tiny.in12k_ft_in1k_384", in_channels=12)

        # DWT 变换模块
        self.dwt = DifferentiableDWT(in_channels=3)

        out_channels = [96, 192, 384, 768]


        # MultiScaleCCA
        self.MultiScaleCCA1 = MultiScaleCCA(out_channels[0])
        self.MultiScaleCCA2 = MultiScaleCCA(out_channels[1])
        self.MultiScaleCCA3 = MultiScaleCCA(out_channels[2])
        self.MultiScaleCCA4 = MultiScaleCCA(out_channels[3])

        self.ECAM1 = EnhancedFreqDCA(out_channels[1])
        self.ECAM2 = EnhancedFreqDCA(out_channels[2])
        self.ECAM3 = EnhancedFreqDCA(out_channels[3])

        # BCAM

        self.BCAM1 = BCAM(out_channels[1])
        self.BCAM2 = BCAM(out_channels[2])
        self.BCAM3 = BCAM(out_channels[3])

        self.decoder4 = DecoderBlock(768, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        self.conv4 = nn.Conv2d(256 + 384, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(128 + 192, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(64 + 96, 64, kernel_size=1)


        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.finalnorm1 = nn.BatchNorm2d(32)
        self.finalrelu1 = nn.ReLU(inplace=True)

        self.finalconv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.finalnorm2 = nn.BatchNorm2d(32)
        self.finalrelu2 = nn.ReLU(inplace=True)

        self.finalconv3 = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # 空间分支输出
        e1_s, e2_s, e3_s, e4_s = self.spatial_backbone(x)

        x_freq = self.dwt(x)
        f1, f2, f3, f4 = self.freq_backbone(x_freq)


        # 空间特征加强
        e1_s = self.MultiScaleCCA1(e1_s)
        e2_s = self.MultiScaleCCA2(e2_s)
        e3_s = self.MultiScaleCCA3(e3_s)
        e4_s = self.MultiScaleCCA4(e4_s)

        # 频率特征加强

        f2 = self.ECAM1(f2)
        f3 = self.ECAM2(f3)
        f4 = self.ECAM3(f4)


        # 空间+频率特征融合
        DSF2 = self.BCAM1(e2_s, f2)
        DSF3 = self.BCAM2(e3_s, f3)
        DSF4 = self.BCAM3(e4_s, f4)

        # 解码
        d4 = self.conv4(torch.cat([self.decoder4(DSF4), DSF3], dim=1))
        d3 = self.conv3(torch.cat([self.decoder3(d4), DSF2], dim=1))
        d2 = self.conv2(torch.cat([self.decoder2(d3), e1_s], dim=1))
        d1 = self.decoder1(d2)

        out1 = self.finalrelu1(self.finalnorm1(self.finaldeconv1(d1)))
        out2 = self.finalrelu2(self.finalnorm2(self.finalconv2(out1)))
        out3 = self.finalconv3(out2)
        out = F.interpolate(out3, size=input_size, mode='bilinear', align_corners=True)

        return out




from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    # input_tensor = torch.randn((1, 3, 512, 512)).to(device)
    model = DSWFNet().to(device)
    summary(model, (1, 3, 1024, 1024))
