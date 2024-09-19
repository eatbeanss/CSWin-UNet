# CSWin-UNet
The codes for the work "CSWin-UNet: Transformer UNet with cross-shaped windows for medical image segmentation". 

## Prepare data
The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd).
## Train/Test

- Train

```bash
python train.py --dataset Synapse --cfg configs/cswin_tiny_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
python test.py --dataset Synapse --cfg configs/cswin_tiny_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer)

## Citation

```bibtex
@article{liu2025cswin,
  title={CSWin-UNet: Transformer UNet with cross-shaped windows for medical image segmentation},
  author={Liu, Xiao and Gao, Peng and Yu, Tao and Wang, Fei and Yuan, Ru-Yue},
  journal={Information Fusion},
  volume={113},
  pages={102634},
  year={2025},
  publisher={Elsevier}
}
```
