# Multi-Faceted Distillation of Base-Novel Commonality for Few-shot Object Detection, ECCV 2022

This repo is built upon [DeFRCN](https://github.com/er-muyue/DeFRCN), where you can download the datasets and the pre-trained weights.


## Requirements
Python == 3.7.10

Pytorch == 1.6.0

Torchvision == 0.7.0

CUDA == 10.1


## File Structure
```
    ├── weight/                   
    |   ├── R-101.pkl              
    |   └── resnet101-5d3b4d8f.pth   
    └── datasets/
        ├── coco/           
        │   ├── annotations/
        │   ├── train2014/
        │   └── val2014/
        ├── cocosplit/
        ├── VOC2007/            
        │   ├── Annotations/
        │   ├── ImageSets/
        │   └── JPEGImages/
        ├── VOC2012/            
        │   ├── Annotations/
        │   ├── ImageSets/
        │   └── JPEGImages/
        └── vocsplit/
```


## Training and Evaluation
* For VOC
```
sh voc_train.sh mfdc SPLIT_ID
```
* For COCO
```
sh coco_train.sh mfdc
```

## Contact
Please feel free to contact me (Email: wushuang9811@outlook.com) if you have any questions.