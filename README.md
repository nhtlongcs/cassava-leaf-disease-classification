# 🌿Cassava Leaf Disease Classification 🌿
### Identify the type of disease present on a Cassava Leaf image

[Contest website](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview/description) 

"As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated.

Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.  

In this competition, we introduce a dataset of 21,367 labeled images collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This is in a format that most realistically represents what farmers would need to diagnose in real life.

Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage." 

_**Quoted from contest decription.**_

## Dataset

![Sample images](img/sample_images.jpg)

**Train set**: 21,367 images.  
**Test set**: ~15,000 images.  
**Public test**: 31% of the test set.  
**Private test**: 69% of the test set.
 
**Classes mapping**

|  Class | Numeric label|
|---|---|
| Cassava Bacterial Blight (CBB)  | 0 |
| Cassava Brown Streak Disease (CBSD)  |  1 | 
|  Cassava Green Mottle (CGM)  | 2  |
|Cassava Mosaic Disease (CMD) |3 |
| Healthy|4|

![classes distribution](img/classes_distribution.png)
> Credit: https://www.kaggle.com/foolofatook/starter-eda-cassava-leaf-disease

## To-do list

- [x] Code baseline and trainer on GPU + TPU  
- [x] Transforms: albumentations
- [x] Implement models: EfficientNet, ViT, Resnext 
- [x] Implement losses: Focal loss, CrossEntropy loss, Bi-Tempered Loss  
- [x] Implement optimizers: SAM  
- [x] Implement schedulers: StepLR, WarmupCosineSchedule  
- [x] Implement metrics: accuracy
- [x] Write inference notebook  
- [x] Implement Stratified K-folding  
- [x] Merge 2019 dataset and 2020 dataset   
- [x] Implement gradient_accumulation   
- [x] Implement Automatic Mixed Precision  
- [x] Write Optuna scripts for hyperparams tuning  
- [x] Evaluate classes distribution of public leaderboard test.  

## Our approach

### Summary

The final model is a stacking of three CNN-based models, [EfficientNet](https://arxiv.org/abs/1905.11946), [ResneXt](https://arxiv.org/abs/1611.05431) and [Densenet](https://arxiv.org/abs/1608.06993).

![final_model](img/final_model.png)

### Models
Most models are from the [Timm library](https://github.com/rwightman/pytorch-image-models).

#### resnext50_32x4d - [training config](src/configs/resnext.yaml)

* Use pretrained weight: Yes
* Freeze backbone: No
* Input image size: 512
* Batch size: 16
* Number of training (max) epochs: 10

#### efficientnet-b4 - [training config](src/configs/effnet.yaml)

* Use pretrained weight: Yes
* Freeze backbone: No
* Input image size: 512
* Batch size: 16
* Number of training (max) epochs: 10

#### densenet121 - [training config](src/configs/densenet.yaml)

* Use pretrained weight: Yes
* Freeze backbone: No
* Input image size: 512
* Batch size: 16
* Number of training (max) epochs: 10
