import torch
from torch import nn
import torch.nn.functional as F
import pywt
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from einops import rearrange

class LayerNorm2d(nn.Module):

    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))  # 可学习缩放系数
        self.bias = nn.Parameter(torch.zeros(num_channels))   # 可学习偏置
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=[2, 3], keepdim=True)  # 在H,W上求均值
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)  # 在H,W上求方差

        x = (x - mean) / torch.sqrt(var + self.eps)  # 标准化
        x = x * self.weight[:, None, None] + self.bias[:, None, None]  # 仿射变换
        return x

class EnhancedFreqDCA(nn.Module):
    def __init__(self, in_channels, reduction=4, drop_path_prob=0.1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        mid_channels = in_channels // reduction

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()

    def forward(self, x):
        b, c, _, _ = x.size()
        gap = self.avgpool(x)  # shape: [b, c, 1, 1]
        gmp = self.maxpool(x)  # shape: [b, c, 1, 1]
        feat = torch.cat([gap, gmp], dim=1)  # shape: [b, 2c, 1, 1]

        weights = self.conv(feat)  # shape: [b, c, 1, 1]
        weights = self.drop_path(weights)
        out = x * weights  # 通道加权
        return out

# DropPath（Stochastic Depth）标准实现
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class BCAM(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, temperature=0.01):
        super(BCAM, self).__init__()
        self.num_heads = num_heads
        self.temperature = temperature

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Gating 权重
        self.gate = nn.Parameter(torch.ones(1))
        self.sigmoid = nn.Sigmoid()

        # Learnable Positional Bias（初始化为 0）
        self.pos_bias_h = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.pos_bias_w = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        # MLP 增强模块
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    def forward(self, x1, x2):
        B, C, h, w = x1.shape

        out1 = self.project_out(x1)
        out2 = self.project_out(x2)

        # KV for vertical
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)

        # KV for horizontal
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)

        # Q for each
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)

        # Cosine normalization
        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        k2 = F.normalize(k2, dim=-1)

        # Cosine Attention + Positional Bias
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) / self.temperature + self.pos_bias_h
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) / self.temperature + self.pos_bias_w

        # Softmax
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        out3 = torch.matmul(attn1, v1) + q1
        out4 = torch.matmul(attn2, v2) + q2

        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Gated 双向融合
        gate_weight = self.sigmoid(self.gate)
        fusion = gate_weight * out3 + (1 - gate_weight) * out4

        # 投影 + MLP增强 + 残差连接
        out = self.project_out(fusion)
        out = self.mlp(out) + x1 + x2

        return out

class MultiScaleCCA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mid_channels = in_channels // reduction
        self.conv1 = nn.Conv2d(in_channels * 2, mid_channels, 1, bias=False)  # 注意通道倍增
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 1, bias=False)

        # 多尺度方向卷积（depthwise）
        self.conv_h_3 = nn.Conv2d(2, mid_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv_h_7 = nn.Conv2d(2, mid_channels, kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.conv_h_11 = nn.Conv2d(2, mid_channels, kernel_size=(1, 11), padding=(0, 5), bias=False)
        self.conv_h_15 = nn.Conv2d(2, mid_channels, kernel_size=(1, 15), padding=(0, 7), bias=False)

        self.conv_v_3 = nn.Conv2d(2, mid_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_v_7 = nn.Conv2d(2, mid_channels, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.conv_v_11 = nn.Conv2d(2, mid_channels, kernel_size=(11, 1), padding=(5, 0), bias=False)
        self.conv_v_15 = nn.Conv2d(2, mid_channels, kernel_size=(15, 1), padding=(7, 0), bias=False)

        self.conv_merge = nn.Conv2d(mid_channels * 8, in_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # -----------------------
        # 1. 通道注意力
        avg = self.avg_pool(x)  # [B, C, 1, 1]
        max = self.max_pool(x)  # [B, C, 1, 1]
        pooled = torch.cat([avg, max], dim=1)  # [B, 2C, 1, 1]
        ca_out = self.conv2(self.relu(self.conv1(pooled)))  # [B, C, 1, 1]

        # -----------------------
        # 2. 坐标注意力方向统计
        h_mean = torch.mean(x, dim=1, keepdim=True)
        h_max = torch.max(x, dim=1, keepdim=True)[0]
        h_base = torch.cat([h_mean, h_max], dim=1)  # [B, 2, H, W]

        v_mean = torch.mean(x, dim=1, keepdim=True)
        v_max = torch.max(x, dim=1, keepdim=True)[0]
        v_base = torch.cat([v_mean, v_max], dim=1)  # [B, 2, H, W]

        # 3. 多尺度方向卷积
        h_out_3 = self.conv_h_3(h_base)
        h_out_7 = self.conv_h_7(h_base)
        h_out_11 = self.conv_h_11(h_base)
        h_out_15 = self.conv_h_15(h_base)

        v_out_3 = self.conv_v_3(v_base)
        v_out_7 = self.conv_v_7(v_base)
        v_out_11 = self.conv_v_11(v_base)
        v_out_15 = self.conv_v_15(v_base)

        # 拼接
        coord_out = torch.cat([
            h_out_3, h_out_7, h_out_11, h_out_15,
            v_out_3, v_out_7, v_out_11, v_out_15
        ], dim=1)
        coord_out = self.conv_merge(coord_out)  # [B, C, H, W]

        # -----------------------
        # 4. 通道注意力 + 空间注意力
        attention = self.sigmoid(ca_out + coord_out)  # [B, C, H, W]

        # -----------------------
        # 5. 残差连接输出
        return self.gamma * (x * attention) + x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(p=0.1)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.dropout(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

class DifferentiableDWT(nn.Module):
    def __init__(self, in_channels, wave='haar', mode='symmetric', normalize=True, dropout_prob=0.2, drop_path_prob=0.1):
        super(DifferentiableDWT, self).__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)
        self.normalize = normalize
        self.in_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "输入尺寸必须为偶数"

        # 小波分解
        LL, Yh = self.dwt(x)  # LL: (B, C, H/2, W/2),  Yh: list with shape (B, C, 3, H/2, W/2)
        LH, HL, HH = Yh[0][:, :, 0], Yh[0][:, :, 1], Yh[0][:, :, 2]

        target_h = H // 2
        target_w = W // 2

        LL = LL[:, :, :target_h, :target_w]
        LH = LH[:, :, :target_h, :target_w]
        HL = HL[:, :, :target_h, :target_w]
        HH = HH[:, :, :target_h, :target_w]

        if self.normalize:
            def norm_fn(tensor):
                mean = tensor.mean(dim=(2, 3), keepdim=True)
                std = tensor.std(dim=(2, 3), keepdim=True) + 1e-6
                return (tensor - mean) / std
            LL = norm_fn(LL)
            LH = norm_fn(LH)
            HL = norm_fn(HL)
            HH = norm_fn(HH)


        IMGF = torch.cat([LL, LH, HL, HH], dim=1)  # [B, 3C, H/2, W/2]

        return IMGF

