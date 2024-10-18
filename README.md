# CLIP-GA
## Dataset
### PASCAL VOC2012
You will need to download the images (JPEG format) in PASCAL VOC2012 dataset from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and train_aug ground-truth can be found [here](http://home.bharathh.info/pubs/codes/SBD/download.html). Make sure your `data/VOC2012 folder` is structured as follows:
```
├── VOC2012/
|   ├── Annotations
|   ├── ImageSets
|   ├── SegmentationClass
|   ├── SegmentationClassAug
|   └── SegmentationObject
```
### MS-COCO 2014
You will need to download the images (JPEG format) in MSCOCO 2014 dataset [here](https://cocodataset.org/#download) and ground-truth mask can be found [here](https://drive.google.com/drive/folders/18l3aAs64Ld_uvAJm57O3EiHuhEXkdwUy?usp=share_link). Make sure your `data/COCO folder` is structured as follows:
```
├── COCO/
|   ├── train2014
|   ├── val2014
|   ├── annotations
|   |   ├── instances_train2014.json
|   |   ├── instances_val2014.json
|   ├── mask
|   |   ├── train2014
|   |   ├── val2014
```

## Training on PASCAL VOC2012
1. Install CLIP.
