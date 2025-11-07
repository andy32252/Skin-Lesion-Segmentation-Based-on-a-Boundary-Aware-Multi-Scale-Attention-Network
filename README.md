# Skin-Lesion-Segmentation-Based-on-a-Boundary-Aware-Multi-Scale-Attention-Network
Download weights from: (Release)
put it into: ./weights/2017_weights.pth

## Inference

1. clone this repo
2.download `2017_weights.pth` from [link](https://drive.google.com/file/d/1ySyrYQ9ZawFjGGX2SHEX9f2W5It9c5w8/view?usp=sharing) and create weight folder, and put it under `./weights/`
3. (optional) dataset: we provide 5 ISIC2017 test images under `datasets/...`
4. run:

```bash
python inference.py --cfg configs/mit_b5_with_glr_fpn.yaml
