# Multi-Faceted Distillation of Base-Novel Commonality for Few-shot Object Detection, ECCV 2022

This repo is built upon [DeFRCN](https://github.com/er-muyue/DeFRCN), where you can download the datasets and the pre-trained weights.


## Requirements
Python == 3.7.10

Pytorch == 1.6.0

Torchvision == 0.7.0

Detectron2 == 0.3

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

## Citation
If you find our code helpful in your research, please cite the following publication:
```
@inproceedings{wu2022multi,
  title={Multi-faceted Distillation of Base-Novel Commonality for Few-Shot Object Detection},
  author={Wu, Shuang and Pei, Wenjie and Mei, Dianwen and Chen, Fanglin and Tian, Jiandong and Lu, Guangming},
  booktitle={European Conference on Computer Vision},
  pages={578--594},
  year={2022},
  organization={Springer}
}
```

## Contact
Please feel free to contact me (Email: wushuang9811@outlook.com) if you have any questions.