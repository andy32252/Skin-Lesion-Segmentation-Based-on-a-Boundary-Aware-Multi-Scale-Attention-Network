import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3x3GNReLU(nn.Module):
    """3x3 convolution + GroupNorm + ReLU, optional upsampling"""
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x

class ReverseAttention(nn.Module):
    """Reverse Attention Module to focus on difficult regions"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ra = 1 - self.sigmoid(x)
        x = x * ra
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class FPNBlock(nn.Module):
    """Feature Pyramid block with skip connection and basic conv layers"""
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        # The skip_conv should convert skip_channels to pyramid_channels
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.conv1 = Conv3x3GNReLU(pyramid_channels, pyramid_channels)
        self.conv2 = Conv3x3GNReLU(pyramid_channels, pyramid_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationBlock(nn.Module):
    """Segmentation branch with configurable upsampling steps"""
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        layers = [Conv3x3GNReLU(in_channels, out_channels, upsample=False)]
        for _ in range(n_upsamples):
            layers.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class MergeBlock(nn.Module):
    """Merge multi-scale features by addition or concatenation"""
    def __init__(self, policy="add"):
        super().__init__()
        if policy not in ["add","cat"]:
            raise ValueError(f"Unknown merge policy: {policy}")
        self.policy = policy

    def forward(self, features):
        if self.policy == "add":
            return sum(features)
        return torch.cat(features, dim=1)

class FPNDecoder(nn.Module):
    """FPN Decoder with Reverse Attention and Deep Supervision"""
    def __init__(
        self,
        encoder_channels,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
        deep_supervision=True
    ):
        super().__init__()
        # ignore first channel and reverse order
        enc = encoder_channels[1:][::-1]
        self.deep_supervision = deep_supervision
        # Store input image shape for upsampling later
        self.input_shape = None
        # Top level P5
        self.p5 = Conv3x3GNReLU(enc[0], pyramid_channels)
        self.ra5 = ReverseAttention(pyramid_channels)
        # Levels P4, P3, P2
        self.p4 = FPNBlock(pyramid_channels, enc[1]); self.ra4 = ReverseAttention(pyramid_channels)
        self.p3 = FPNBlock(pyramid_channels, enc[2]); self.ra3 = ReverseAttention(pyramid_channels)
        self.p2 = FPNBlock(pyramid_channels, enc[3]); self.ra2 = ReverseAttention(pyramid_channels)
        # Segmentation branches
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n)
            for n in [3,2,1,0]
        ])
        self.merge = MergeBlock(merge_policy)
        merged_ch = segmentation_channels * (4 if merge_policy=="cat" else 1)
        self.dropout = nn.Dropout2d(p=dropout)
        self.final_head = nn.Conv2d(merged_ch, 1, kernel_size=1)
        if self.deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(segmentation_channels,1,kernel_size=1)
                for _ in range(len(self.seg_blocks))
            ])

    def forward(self, *features):
        # Store input resolution for final upsampling
        self.input_shape = features[0].shape[2:]
        
        # expect at least 5 features; take last five
        c1, c2, c3, c4, c5 = features[-5:]
        # P5
        p5 = self.p5(c5); p5 = self.ra5(p5)
        # P4
        p4 = self.p4(p5, c4); p4 = self.ra4(p4)
        # P3
        p3 = self.p3(p4, c3); p3 = self.ra3(p3)
        # P2
        p2 = self.p2(p3, c2); p2 = self.ra2(p2)
        # Build pyramid
        pyramid = [p5, p4, p3, p2]
        seg_feats = [blk(p) for blk, p in zip(self.seg_blocks, pyramid)]
        x = self.merge(seg_feats)
        x = self.dropout(x)
        main_out = self.final_head(x)
        
        # Upsample main output to match input resolution
        main_out = F.interpolate(main_out, size=self.input_shape, mode="bilinear", align_corners=True)
        
        if self.deep_supervision:
            aux_outs = []
            for head, f in zip(self.ds_heads, seg_feats):
                # Apply head to get segmentation output
                aux_out = head(f)
                # Upsample to match input resolution
                aux_out = F.interpolate(aux_out, size=self.input_shape, mode="bilinear", align_corners=True)
                aux_outs.append(aux_out)
            return main_out, aux_outs
        return main_out