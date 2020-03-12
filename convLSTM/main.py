import pandas as pd
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

# from transforms import strong_aug, base_transform
from collections import defaultdict
import torch
from itertools import chain

# from model import get_model
from dataloader import Dataloader
from utils import train_model, read_json, read_dataset, make_dict
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import Counter
from config import get_config, get_aug_config
from augmentation import get_aug
from model import ConvLSTM
import wandb
import json
import glob
from RAdam.radam.radam import RAdam

# import torchvision


# config and meta files
cfg, aug_cfg = get_config(), get_aug_config()
cfg = dict(chain(cfg.items(), aug_cfg.items()))


meta = read_json("./train_sample_videos/metadata.json")


# wandb
wandb.init(project="DFDC_convlstm")
wandb.config = cfg


# if cfg["data"] == "total":
#     # images1, labels1 = read_dataset('/hdd2/dfdc_dataset/4_dataframes1')
#     images2, labels2 = read_dataset("/hdd2/dfdc_dataset/4_dataframes2")
#     total_images = images2
#     total_labels = labels2
#     N = len(total_images)
#     train_paths, train_labels, val_paths, val_labels = (
#         total_images[: int(N * 0.9)],
#         total_labels[: int(N * 0.9)],
#         total_images[int(N * 0.9) :],
#         total_labels[int(N * 0.9) :],
#     )

# else:
#     train_video_cropped = glob.glob("./train_crop/*")
#     val_video_cropped = glob.glob("./test_crop/*")

#     train_paths, train_labels = [], []
#     val_paths, val_labels = [], []

#     for v in train_video_cropped:
#         v_name = v.split("/")[-1]
#         label = 0 if meta[v_name]["label"] == "REAL" else 1
#         train_labels.append(label)
#         train_paths.append(v)

#     train_paths, train_labels, val_paths, val_labels = (
#         train_paths[:300],
#         train_labels[:300],
#         train_paths[300:],
#         train_labels[300:],
#     )

# print("train: {} val: {}".format(len(train_paths), len(val_paths)))


if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")
    # Define dataloader
    transform = get_aug(cfg)
    # train
    class_dict = np.load("./dict/train_class_dict.npy", allow_pickle=True).item()
    length_dict = np.load("./dict/train_length_dict.npy", allow_pickle=True).item()
    class_idx = [-1, -1]

    train_dataset = Dataloader(
        class_dict, length_dict, class_idx, True, cfg, transform=transform
    )

    # val
    class_dict = np.load("./dict/val_class_dict.npy", allow_pickle=True).item()
    length_dict = np.load("./dict/val_length_dict.npy", allow_pickle=True).item()
    class_idx = [-1, -1]

    val_dataset = Dataloader(class_dict, length_dict, class_idx, False, cfg)
    imgs, labels = train_dataset.__getitem__(0)

    print("-----Sample-----")
    print("img: ", imgs.shape)
    print("labels: ", labels.shape)
    print()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    dataloaders = {}
    dataloaders["train"] = train_dataloader
    dataloaders["val"] = val_dataloader

    # Define Model
    device = torch.device("cuda")

    model = ConvLSTM(cfg, bias=True, batch_first=True)

    if cfg["resume"]:
        model.load_state_dict(torch.load(cfg["weights"]))

    model.to(device)

    if cfg["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=cfg["weight_decay"],
        )
    else:
        optimizer = RAdam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )

    criterion = nn.BCELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    train_model(model, criterion, dataloaders, optimizer, exp_lr_scheduler, device, 100)

