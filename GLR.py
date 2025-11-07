import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio=0.25, pooling_type='att', fusion_types=('channel_add', 'channel_mul')):
        super(ContextBlock, self).__init__()
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def forward(self, x):
        batch, channel, height, width = x.size()

        if self.pooling_type == 'att':
            input_x = x.view(batch, channel, -1)
            context_mask = self.conv_mask(x).view(batch, 1, -1)
            context_mask = self.softmax(context_mask)
            context = torch.matmul(input_x, context_mask.permute(0, 2, 1)).view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvBranch(nn.Module):
    def __init__(self, in_features, embed_dim):
        super(ConvBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_features, embed_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False, groups=embed_dim)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        return x

class GLoR(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(GLoR, self).__init__()
        self.local_branch = ConvBranch(input_dim, embed_dim)
        self.global_branch = ContextBlock(input_dim, ratio=0.25)
        self.fusion_conv = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, x):
        local_feat = self.local_branch(x)  # Local feature extraction
        global_feat = self.global_branch(x)  # Global feature extraction

        # Concatenate and fuse
        combined_feat = torch.cat([local_feat, global_feat], dim=1)
        output = self.fusion_conv(combined_feat)
        return output

class BGAM(nn.Module):#Boundary-Guided Attention Module；GAF
    def __init__(self, threshold=0.5):
        super(BGAM, self).__init__()
        self.threshold = threshold  # 設定語義區域的適應性閾值
        self.conv1x1 = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.conv1x1.weight, 1)
        nn.init.constant_(self.conv1x1.bias, 0)

    def forward(self, semantic_score, boundary_score):
        semantic_similarity = F.cosine_similarity(semantic_score, semantic_score, dim=1)
        boundary_similarity = F.cosine_similarity(boundary_score, boundary_score, dim=1)

        semantic_tables = (semantic_similarity > self.threshold).float()
        boundary_tables = (boundary_similarity > self.threshold).float()

        region_attention_tables = (1 - semantic_tables) * (1 - boundary_tables)
        region_decision_tables = (region_attention_tables + region_attention_tables.permute(0, 2, 1)) / 2

        return region_attention_tables, region_decision_tables

class AGCR(nn.Module):#Attention-Guided Context Refinement；DIR
    def __init__(self, in_channels=512, out_channels=512, top_k_ratio=0.1):
        super(AGCR, self).__init__()
        self.top_k_ratio = top_k_ratio
        self.query_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)

    def forward(self, features, region_attention_tables):
        batch, channels, height, width = features.size()
        num_regions = int(height * width * self.top_k_ratio)

        query = self.query_conv(features).view(batch, -1, height * width)
        key = self.key_conv(features).view(batch, -1, height * width)
        value = self.value_conv(features).view(batch, channels, height * width)

        attention_map = torch.bmm(query.permute(0, 2, 1), key) / torch.sqrt(torch.tensor(query.size(1)).float())
        attention_weights = F.softmax(attention_map, dim=-1)

        soft_top_k = torch.topk(attention_weights, num_regions, dim=-1)[0].mean(dim=-1)
        attention_weights = attention_weights * soft_top_k.unsqueeze(1)

        context = torch.bmm(value, attention_weights.permute(0, 2, 1))
        context = context.mean(dim=-1).view(batch, channels, 1, 1)
        context = context.expand(-1, -1, height, width)

        # **確保 region_attention_tables 是 (batch, 1, H, W)**
        if len(region_attention_tables.shape) == 3:
            region_attention_tables = region_attention_tables.unsqueeze(1)  # 變成 (batch, 1, H, W)

        elif region_attention_tables.shape[1] != channels:
            # 如果 `region_attention_tables` channel 數不匹配，用 `Conv2d` 調整
            region_attention_tables = self.region_conv(region_attention_tables)

        refined_features = torch.cat([features, context * region_attention_tables], dim=1)
        output = self.final_conv(refined_features)

        return output

class ReverseAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReverseAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, attention_map):
        reverse_map = 1 - attention_map  # Reverse attention operation
        weighted_x = x * reverse_map  # Apply reverse attention
        out = F.relu(self.bn1(self.conv1(weighted_x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class GRAM(nn.Module):#GLR_module
    def __init__(self, in_channels, embed_dim, num_classes):
        super(GRAM, self).__init__()
        self.glsa = GLoR(in_channels, embed_dim)  # Global-Local region 
        self.semantic_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(embed_dim, 2, kernel_size=1)
        self.gaf = BGAM(num_classes)  # BISF
        self.dir = AGCR(embed_dim)  # AGCR
        self.reverse_attention = ReverseAttention(2, 2)  # Add RA Module
    def forward(self, x):
        x = self.glsa(x)  # Extract global-local features
        semantic_logits = self.semantic_head(x)
        boundary_logits = self.boundary_head(x)
        # Apply Reverse Attention on boundary logits
        refined_boundary = self.reverse_attention(boundary_logits, torch.sigmoid(boundary_logits))
        region_attention, _ = self.gaf(semantic_logits, refined_boundary)
        x = self.dir(x, region_attention)
        return x