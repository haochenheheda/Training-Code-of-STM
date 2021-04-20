# STM_training
This repository fully reproduces this work: [Space-Time Memory Networks](https://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html)

## Introduction

## Requirements
- Python >= 3.6
- Pytorch 1.5
- Numpy
- Pillow
- opencv-python
- imgaug
- scipy
- tqdm
- pandas

## Datasets

#### [MS-COCO](https://cocodataset.org/#home)
#### [DAVIS](https://davischallenge.org/)
#### [Youtube-VOS](https://youtube-vos.org/)

#### Structure
```
 |- data
      |- Davis
          |- JPEGImages
          |- Annotations
          |- ImageSets
      
      |- Youtube-vos
          |- train
          |- valid
      |- Ms-COCO
          |- train2017
          |- annotations
              |- instances_train2017.json
```

## Training

#### Stage 1
Pretraining on MS-COCO
```
python train_coco.py -Ddavis "path to davis" -Dcoco "path to coco" -save "path to checkpoints"
```

#### Stage 2
Training on Davis&Youtube-vos
```
python train_davis.py -Ddavis "path to davis" -Dyoutube "path to youtube-vos" -save "path to checkpoints" -resume "path to coco pretrained weights"
```
## Performance&Weights

|  | backbone |  training stage | J&F | J |  F  | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Ours| resnet-50 |  stage 1 | 69.5 | 67.8 | 71.2 | xx |
| Origin | resnet-50 | stage 2 | 81.8 | 79.2 | 84.3 | [`link`](https://github.com/seoungwugoh/STM) |
| Ours| resnet-50 | stage 2 | 82.0 | 79.7 | 84.4 | xx |

## Notes

## Citing STM
@inproceedings{oh2019video,
  title={Video object segmentation using space-time memory networks},
  author={Oh, Seoung Wug and Lee, Joon-Young and Xu, Ning and Kim, Seon Joo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9226--9235},
  year={2019}
}
