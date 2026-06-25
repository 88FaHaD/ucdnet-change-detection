import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs=2):
        super().__init__()
        convs = [ConvBnRelu(in_ch, out_ch)]
        for _ in range(num_convs - 1):
            convs.append(ConvBnRelu(out_ch, out_ch))
        self.convs = nn.Sequential(*convs)
        self.residual_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1)

    def forward(self, x1, x2):
        f1 = self.convs(x1)
        f2 = self.convs(x2)
        diff = self.residual_conv(F.relu(f1 - f2))
        out1 = torch.cat([f1, diff], dim=1)  # (B, 2*out_ch, H, W)
        out2 = torch.cat([f2, diff], dim=1)
        return out1, out2, f1, f2


class PoolingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.strided_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                      stride=stride, padding=1)
        self.avg_pool  = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        sc = F.relu(self.strided_conv(x))
        ap = F.relu(self.pointwise(self.avg_pool(x)))
        return sc + ap


class GlobalPoolingBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, target_h, target_w):
        g = x.mean(dim=[2, 3], keepdim=True)       # (B, C, 1, 1)
        g = F.relu(self.pointwise(g))
        g = F.interpolate(g, size=(target_h, target_w),
                          mode='bilinear', align_corners=False)
        return g


class NSPPBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        mid_ch = in_ch // 4

        self.pool1  = PoolingBlock(in_ch, mid_ch, stride=2)
        self.pool2  = PoolingBlock(in_ch, mid_ch, stride=4)
        self.pool3  = PoolingBlock(in_ch, mid_ch, stride=8)

        self.gpool1 = GlobalPoolingBlock(mid_ch, mid_ch)
        self.gpool2 = GlobalPoolingBlock(mid_ch, mid_ch)
        self.gpool3 = GlobalPoolingBlock(mid_ch, mid_ch)

        # 3*mid_ch + in_ch → in_ch
        self.final_conv = nn.Conv2d(3 * mid_ch + in_ch, in_ch, kernel_size=1)

    def forward(self, x):
        h, w    = x.shape[2], x.shape[3]
        mid_ch  = x.shape[1] // 4

        p1 = self.pool1(x)                          # (B, mid_ch, h/2, w/2)
        p2 = self.pool2(x)                          # (B, mid_ch, h/4, w/4)
        p3 = self.pool3(x)                          # (B, mid_ch, h/8, w/8)

        p1 = self.gpool1(p1, h, w)                  # (B, mid_ch, h, w)
        p2 = self.gpool2(p2, h, w)
        p3 = self.gpool3(p3, h, w)

        out = torch.cat([p1, p2, p3, x], dim=1)     # (B, 3*mid_ch + in_ch, h, w)
        return F.relu(self.final_conv(out))          # (B, in_ch, h, w)


class UCDNet(nn.Module):
    def __init__(self, in_channels=13, num_classes=2):
        super().__init__()

        # Encoder stages
        # Stage 1: in=13  → out=16,  cat_out=32
        # Stage 2: in=32  → out=32,  cat_out=64
        # Stage 3: in=64  → out=64,  cat_out=128
        # Stage 4: in=128 → out=128, cat_out=256
        self.enc1 = EncoderStage(in_channels, 16,  num_convs=2)
        self.enc2 = EncoderStage(32,          32,  num_convs=2)
        self.enc3 = EncoderStage(64,          64,  num_convs=3)
        self.enc4 = EncoderStage(128,         128, num_convs=3)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck: f1(128) + diff(128) = 256 → 128
        self.bottleneck_conv = nn.Conv2d(128 + 128, 128, kernel_size=1)

        # NSPP
        self.nspp = NSPPBlock(128)

        # Decoder
        # up1: 128 → 64, skip: enc3 f1(64) + f2(64) = 128, total = 64+64+64=192
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBnRelu(64 + 64 + 64, 64),
            ConvBnRelu(64, 64),
            ConvBnRelu(64, 32)
        )

        # up2: 32 → 32, skip: enc2 f1(32) + f2(32) = 64, total = 32+32+32=96
        self.up2  = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBnRelu(32 + 32 + 32, 32),
            ConvBnRelu(32, 16)
        )

        # up3: 16 → 16, skip: enc1 f1(16) + f2(16) = 32, total = 16+16+16=48
        self.up3  = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBnRelu(16 + 16 + 16, 16)
        )

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x1, x2):
        # Encoder
        e1_out1, e1_out2, e1_f1, e1_f2 = self.enc1(x1, x2)
        # e1_f: (B,16,64,64)  e1_out: (B,32,64,64)
        p1_1 = self.pool(e1_out1)
        p1_2 = self.pool(e1_out2)

        e2_out1, e2_out2, e2_f1, e2_f2 = self.enc2(p1_1, p1_2)
        # e2_f: (B,32,32,32)  e2_out: (B,64,32,32)
        p2_1 = self.pool(e2_out1)
        p2_2 = self.pool(e2_out2)

        e3_out1, e3_out2, e3_f1, e3_f2 = self.enc3(p2_1, p2_2)
        # e3_f: (B,64,16,16)  e3_out: (B,128,16,16)
        p3_1 = self.pool(e3_out1)
        p3_2 = self.pool(e3_out2)

        e4_out1, e4_out2, e4_f1, e4_f2 = self.enc4(p3_1, p3_2)
        # e4_f: (B,128,8,8)

        # Bottleneck
        diff4      = torch.abs(e4_f1 - e4_f2)              # (B,128,8,8)
        bottleneck = torch.cat([e4_f1, diff4], dim=1)       # (B,256,8,8)
        bottleneck = F.relu(self.bottleneck_conv(bottleneck))# (B,128,8,8)

        # NSPP
        nspp_out = self.nspp(bottleneck)                     # (B,128,8,8)

        # Decoder
        d1 = self.up1(nspp_out)                              # (B,64,16,16)
        d1 = torch.cat([d1, e3_f1, e3_f2], dim=1)           # (B,192,16,16)
        d1 = self.dec1(d1)                                   # (B,32,16,16)

        d2 = self.up2(d1)                                    # (B,32,32,32)
        d2 = torch.cat([d2, e2_f1, e2_f2], dim=1)           # (B,96,32,32)
        d2 = self.dec2(d2)                                   # (B,16,32,32)

        d3 = self.up3(d2)                                    # (B,16,64,64)
        d3 = torch.cat([d3, e1_f1, e1_f2], dim=1)           # (B,48,64,64)
        d3 = self.dec3(d3)                                   # (B,16,64,64)

        out = self.final(d3)                                 # (B,2,64,64)
        return out


# Test with dummy input
model  = UCDNet(in_channels=13, num_classes=2)
dummy1 = torch.randn(2, 13, 64, 64)
dummy2 = torch.randn(2, 13, 64, 64)
out    = model(dummy1, dummy2)

print("Model output shape:", out.shape)   # Expected: (2, 2, 64, 64)
print("Total parameters: {:,}".format(sum(p.numel() for p in model.parameters())))