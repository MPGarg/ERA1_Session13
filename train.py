"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm

from pytorch_lightning import LightningModule, Trainer, seed_everything
import torch 
from torchmetrics import Accuracy
import torch.optim as optim
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import pandas as pd 
from IPython.core.display import display
import seaborn as sn
import os

from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        #if config.SAVE_MODEL:
        #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()

class YOLOTraining(LightningModule):
    def __init__(self,loss_fn,config,model,max_lr,train_loader,pct_start):
        super().__init__()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.config = config
        self.loss_fn = loss_fn
        self.scaled_anchors = (torch.tensor(self.config.ANCHORS)
                                * torch.tensor(self.config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
                            ).to(self.config.DEVICE)
        self.model = model
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.train_loader = train_loader
        
    def forward(self, x):
        return self.model(x)
    
    def check_accuracy(self,x,y,model,threshold,device):        
        x = x.to(device)
        with torch.no_grad():
            out = model(x)
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0

        for i in range(3):
            y[i] = y[i].to(device)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    def training_step(self, batch, batch_id):
        x,y=batch
        x = x.to(self.config.DEVICE)
        y0, y1, y2 = (
            y[0].to(self.config.DEVICE),
            y[1].to(self.config.DEVICE),
            y[2].to(self.config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = self(x)
            loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )

            self.log(f"train_loss", loss, prog_bar=True)

        return loss
    
    def evaluate(self, batch,stage=None):
        x,y=batch
        x = x.to(self.config.DEVICE)
        y0, y1, y2 = (
            y[0].to(self.config.DEVICE),
            y[1].to(self.config.DEVICE),
            y[2].to(self.config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = self(x)
            loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )
            val_accuracy = self.check_accuracy(x,y,self.model,self.config.CONF_THRESHOLD,self.config.DEVICE)
            if val_accuracy!=None:
                clas_acc,no_obj_acc,obj_acc = val_accuracy[0],val_accuracy[1],val_accuracy[2]
                self.log(f"{stage}_loss", loss, prog_bar=True)
                self.log(f"{stage}_clasacc", clas_acc, prog_bar=True)
                self.log(f"{stage}_no_obj_acc", no_obj_acc, prog_bar=True)
                self.log(f"{stage}_obj_acc", obj_acc, prog_bar=True)
            else:
                self.log(f"{stage}_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
      optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.LEARNING_RATE/10, weight_decay=self.config.WEIGHT_DECAY
        )
      scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                steps_per_epoch=len(self.train_loader),
                epochs=self.config.NUM_EPOCHS,
                pct_start=self.pct_start,
                div_factor=10,
                three_phase=False,
                final_div_factor=10,
                anneal_strategy='linear',verbose=False),
      }
      return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    
if __name__ == "__main__":
    main()