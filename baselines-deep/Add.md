# Some changes into this repo

Beyond [pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation/tree/master), we add more models and a new dataset (with its dataloader) to deal with agricultural field delineation task in deep learning methods.

## Models

The additional models are:

- [BsiNet](https://github.com/long123524/BsiNet-torch)
- [SEANet](https://github.com/long123524/SEANet_torch)
- [REAUNet](https://github.com/Remote-Sensing-of-Land-Resource-Lab/REAUNet)

## Datasets

We add [2020 CCF BDCI](https://aistudio.baidu.com/datasetdetail/55400) dataset. This is a multiclass semantic segmentation dataset (7 categories) including "cropland", which makes us eazy to transfer into simple segmentation task (cropland / background).
