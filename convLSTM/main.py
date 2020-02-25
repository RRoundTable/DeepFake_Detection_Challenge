import pandas as pd
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

# from transforms import strong_aug, base_transform
from collections import defaultdict
import torch

# from model import get_model
from dataloader import Dataloader
from utils import train_model, read_json
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import Counter
from config import get_config
from model import ConvLSTM
import wandb
import json
import glob
import torchvision


# config and meta files
cfg = get_config()
meta = read_json(
    "./train_sample_videos/metadata.json"
)


# wandb
wandb.init(project="DFDC_convlstm")
wandb.config = cfg


train_video_cropped = glob.glob("./train_crop/*")
val_video_cropped = glob.glob("./test_crop/*")

train_paths, train_labels = [], []
val_paths, val_labels = [], []

for v in train_video_cropped:
    v_name = v.split("/")[-1]
    label = 0 if meta[v_name]["label"] == "REAL" else 1
    train_labels.append(label)
    train_paths.append(v)

# for v in val_video_cropped:
#     v_name = v.split("/")[-1]
#     label = 0 if meta[v_name]["label"] == "REAL" else 1
#     val_labels.append(label)
#     val_paths.append(v)

train_paths, train_labels, val_paths, val_labels = train_paths[:300], train_labels[:300], train_paths[300:], train_labels[300:]

print('train: {} val: {}'.format(len(train_paths), len(val_paths)))

if __name__ == "__main__":

    # Define dataloader
    train_dataset = Dataloader(train_paths, train_labels, cfg["seq_len"], cfg['mask'], size=cfg['input_size'])
    val_dataset = Dataloader(val_paths, val_labels, cfg["seq_len"], cfg['mask'], size=cfg['input_size'])
    imgs, labels = train_dataset.__getitem__(0)

    print("-----Sample-----")
    print("img: ", imgs.shape)
    print("labels: ", labels.shape)
    print()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['batch_size'], shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg['batch_size'], shuffle=False
    )
    dataloaders = {}
    dataloaders["train"] = train_dataloader
    dataloaders["val"] = val_dataloader

    # Define Model
    device = torch.device("cuda")
    model = ConvLSTM(
        input_size=cfg["input_size"],
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        kernel_size=cfg["kernel_size"],
        bias=True,
        num_layers=cfg["num_layers"],
        seq_len=cfg["seq_len"],
        batch_first=True,
    )
    model.to(device)
    
    if cfg["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=cfg["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=cfg["weight_decay"],
        )

    criterion = nn.BCELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    train_model(model, criterion, dataloaders, optimizer, exp_lr_scheduler, device, 100)

