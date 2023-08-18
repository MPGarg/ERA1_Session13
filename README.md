# Session-13 Assignment

## Problem Statement

![image](https://github.com/MPGarg/ERA1_Session13/assets/120099863/6879ef1b-b8f2-46bc-a9a0-41cd304730c0)

## Code Details

All classes & functions are defined in [link](https://github.com/MPGarg/ERA1_Session13). 

In notebook ERA1_S13.ipynb [link](ERA1_S13.ipynb) functions from the main repository are called. YOLOV3 is trainined using PyTorch-Lightning.

Model was trained for 40 epochs using OneCycle Scheduler. 

**Accuracy for model:**
Class accuracy is: 78.290955%
No obj accuracy is: 97.919060%
Obj accuracy is: 65.142708%

### Accuracy & Loss Curve

![image](https://github.com/MPGarg/ERA1_Session13/assets/120099863/2eeb489d-0eb3-40d4-83c7-b5cfa2cc18f2)

## Hugging face App 

App is created on hugging face for this model:
https://huggingface.co/spaces/MadhurGarg/YOLOV3_PyTorch_Lightning

## Conclusion

* Pytorch-lightning reduces effort on training the model.
* Augmented dataset can be passed in data module and it will create loader for it.
* We need not to mention device on which model needs to be trained. It is taken care by pytorch-lightning
* Hugging face app helps us visualize results and make model presentable
