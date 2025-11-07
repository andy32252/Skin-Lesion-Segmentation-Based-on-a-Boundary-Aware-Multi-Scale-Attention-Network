import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.amp import autocast

class ISIC_2017_Seg_Model(nn.Module):
    def __init__(self, encoder_name, encoder_weights, decoder_name):
        super(ISIC_2017_Seg_Model, self).__init__()
        if decoder_name == 'UnetPlusPlus':
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
        elif decoder_name == 'FPN':
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
        elif decoder_name == 'DeepLabV3':
            self.model = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
        else:
            raise ValueError("decoder_name")
    
    def forward(self, x):
        with autocast(device_type='cuda'):
            return self.model(x)
        
    def save_model_architecture(self, filepath="model_GLR_architecture.txt"):
        with open(filepath, "w") as f:
            f.write(str(self))
            print(f"Model architecture saved to {filepath}")