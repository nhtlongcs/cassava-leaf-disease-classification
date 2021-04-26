# **Cassava Leaf Disease Classification source code**

## **Requirements**

Python 3.8 or later with all [requirements.txt](./requirements.txt) dependencies installed. To install run:
```bash
$ pip install -r requirements.txt
```


## **Environments**
This repo may be run on these verified environments (with all dependencies preinstalled):

- **Google Colab Notebook** 

    <a href="link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## **Organize Directories**

```
  this repo
  └─── src                        
  │    └─── data                    # Dataset folder               
  │         └─── info.csv
  │         └─── train_images
  │                     ***.png
  │    └─── configs                 # Config folder                                          
  │         └─── cfg.yaml
  |              
  │    └─── lists                   # labels folder               
  │         └─── train.csv
  │         └─── val.csv
  │         └─── folds
  │              └─── train
  │                     train#.csv
  │              └─── val
  │                     val#.csv
  │    └─── runs                    # experiments folder               
  │         └─── run_id
  │              └─── checkpoint_fold#
  │                 └─── best_metric_*.pth
  |    train_base.py      
  |    ...                  
  readme.md
  ...
```
## **How to use yaml config (Edit in configs/\*.yaml file):**
Setting config for classification model
```
id:                                 <experiment name>
pretrained:                         <path/to/pretrained>
model:
    name:                           <model class name>
    args:
        <argument keyword>:         <argument value>
        ...
optimizer:
    name:                           <optimizer class name>
    args:
        <argument keyword>:         <argument value>
        ...
loss:
    name:                           <loss class name> 
    args:
        <argument keyword>:         <argument value>
        ...
metric:
    -   name:                       <metric class name> 
        args:
            <argument keyword>:     <argument value>
            ...
    -   ...
scheduler:
    name:                           <scheduler class name>
    args:
        <argument keyword>:         <argument value>
        ...
dataset:
    num_folds:                      <argument value>
    folds_train_dir:                <path/to/csv_train_folder>
    folds_test_dir:                 <path/to/csv_val_folder>
    train:
        name:                       <dataset class name>
        args:
            <argument keyword>:     <argument value>
            ...
        loader:
            name:                   <dataloader class name>
            args:
                <argument keyword>: <argument value>
                ...
    val:
        name:                       <dataset class name>
        args:
            <argument keyword>:     <argument value>
            ...
        loader:
            name:                   <dataloader class name>
            args:
                <argument keyword>: <argument value>
                ...
trainer: 
    nepochs:                        <max number of epochs>
    val_step:                       <validation interval>
    log_step:                       <training log interval>
```
## **Inference**

infer.py runs inference ... , downloading models automatically from the ... and saving results to ...
To run inference on example images in ...:
```bash
$ python infer.py --weights 'best.pth' \
--img 512 \
--source $TEST_DIR \
```
## **Training**

Run commands below to to something
```bash
$ python train_base.py --config configs/densenet.yaml --gpus 0 --fp16 --verbose
```
## **Hyperparameters search**

See [Optuna](optuna/).
