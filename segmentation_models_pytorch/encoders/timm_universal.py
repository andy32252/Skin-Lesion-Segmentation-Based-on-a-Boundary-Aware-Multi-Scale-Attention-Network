# segmentation_models_pytorch/encoders/timm_universal.py
import timm
import torch
import torch.nn as nn

class TimmUniversalEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        # 建 timm backbone
        self.model = timm.create_model(
            name,
            features_only=True,          # timm 會回傳多個stage feature
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )
        self._out_channels = [f["num_chs"] for f in self.model.feature_info]
        self._depth = depth

        # 改輸入通道數
        if in_channels != 3:
            first_conv = self.model.conv_stem if hasattr(self.model, "conv_stem") else self.model.model.conv1
            weight = first_conv.weight
            new_conv = nn.Conv2d(
                in_channels,
                weight.shape[0],
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=(first_conv.bias is not None),
            )
            # 簡單初始化
            with torch.no_grad():
                new_conv.weight[:] = weight.mean(dim=1, keepdim=True)
            if hasattr(self.model, "conv_stem"):
                self.model.conv_stem = new_conv
            else:
                self.model.model.conv1 = new_conv

        # 這裡先不做 make_dilated，有需要再補
        self.output_stride = output_stride

    def forward(self, x):
        # 回傳 list: [F1, F2, ..., Fdepth]
        features = self.model(x)
        return features

    @property
    def out_channels(self):
        return self._out_channels

    def set_in_channels(self, in_channels, pretrained=True):
        # 上面 __init__ 已經處理過了，這裡可以留空或再做一次
        pass

    def make_dilated(self, output_stride):
        # 若之後要支援 16/8 再實作
        self.output_stride = output_stride
