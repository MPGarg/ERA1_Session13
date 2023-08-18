# Session-12 Assignment

## Problem Statement

![image](https://github.com/MPGarg/ERA1_Session12/assets/120099863/192fce1e-ccad-41c6-97ad-4a4bebc8eee8)

## Code Details

All classes & functions are defined in [link](https://github.com/MPGarg/ERA1_main_repo). 

In notebook ERA1_S12.ipynb [link](ERA1_S12.ipynb) functions from the main repository are called. Custom Resnet (Session-10) is trainined using PyTorch-Lightning.

pl.LightningDataModule is inherited to implement our augmentation on dataset. Following funtions of class are implemented:
* __init__
* prepare_data
* setup
* train_dataloader
* val_dataloader
* test_dataloader

For LightningModule following functions are implemented:
* __init__
* forward
* training_step
* evaluate
* validation_step
* test_step
* configure_optimizers

Model was trained for 24 epochs using OneCycle Scheduler. Test accuracy of 91.67% was achieved by end of training.

### Accuracy & Loss Curve

![image](https://github.com/MPGarg/ERA1_Session12/assets/120099863/98c58626-1f26-40fe-8fff-9496668b6595)

### Misclassified Images

![image](https://github.com/MPGarg/ERA1_Session12/assets/120099863/5cf339a3-d7c3-480d-ab82-268837ab17d2)

## Hugging face App 

App is created on hugging face for this model:
https://huggingface.co/spaces/MadhurGarg/PyTorch-Lightning

## Conclusion

* Pytorch-lightning reduces effort on training the model.
* Augmented dataset can be passed in data module and it will create loader for it.
* We need not to mention device on which model needs to be trained. It is taken care by pytorch-lightning
* Hugging face app helps us visualize results and make model presentable
