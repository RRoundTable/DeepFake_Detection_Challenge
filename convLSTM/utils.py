from sklearn.metrics import precision_score, recall_score, accuracy_score
import time
import torch
from torch import nn
import wandb
from config import get_config
import imageio
import numpy as np
import pandas as pd
import cv2
import json

configs = get_config()

def train_model(model, criterion, dataloaders, optimizer, scheduler, device, num_epochs):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        logs = {}
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # save targets and predictions
            targets = []
            predictions = []
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                if inputs.size()[1] < configs['seq_len']: continue
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float() * 1
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
    
                targets += list(labels.detach().cpu().numpy())
                predictions += list(preds.detach().cpu().numpy())
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
              
                
            if phase == "train":
                scheduler.step()
            # metrics
            epoch_loss = running_loss / (len(dataloaders[phase]) * dataloaders[phase].batch_size)
            epoch_acc = accuracy_score(targets, predictions)
            epoch_recall_1 = recall_score(
                targets, predictions, average=None
            )[0]
            epoch_prec_1 = precision_score(
                targets, predictions, average=None
            )[0]
      
            # write logs
            logs[phase + "_loss"] = epoch_loss
            logs[phase + "_acc"] = epoch_acc
            logs[phase + "_recall_1"] = epoch_recall_1
            logs[phase + "_prec_1"] = epoch_prec_1

          
            print(
                "{} Loss: {:.4f} Acc: {:.4f} rec1: {:.4f} prec1: {:.4f}".format(
                    phase,
                    epoch_loss,
                    epoch_acc,
                    epoch_recall_1,
                    epoch_prec_1
                )
            )
            
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
                path = './weights/{}_wd_{}_drop_{}_epoch_{}_acc_{:.4f}_prec1_{:.4f}_recall1_{:.4f}.pth'.format(
                    configs['model'],
                    configs['weight_decay'],
                    configs['dropout'],
                    epoch,
                    epoch_acc,
                    epoch_prec_1,
                    epoch_recall_1
                )
                torch.save(model.state_dict(), path)
        print()

        wandb.log(logs)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))


def test_model(model, testfiles, device, transform, path):
    model.to(device)
    model.eval()

    # save targets and predictions
    results = []
    for path in testfiles:
        img = imageio.imread(path)
        img = transform(image=img)['image']
        img = np.transpose(img, (2, 1, 0))
        img = np.expand_dims(img, axis=0)
        img = torch.as_tensor(img, dtype=torch.float32)
        img = img.to(device)
        outputs = model(img)
        _, pred = torch.max(outputs, -1)
        results += [list(pred.detach().cpu().numpy())[0] + 1]
    print("--- inference is done! ---")
    print()
    df = pd.DataFrame(results)
    df.columns = ['prediction']
    df.to_csv(path, index=False)
    df.to_csv('/job_fair/submit.csv', index=False)
    print(df)

def test_ensemble_model(model1, model2, testfiles, device, transform, path):
    model1.to(device)
    model1.eval()

    model2.to(device)
    model2.eval()

    # save targets and predictions
    results = []
    for path in testfiles:
        img = imageio.imread(path)
        img = transform(image=img)['image']
        img = np.transpose(img, (2, 1, 0))
        img = np.expand_dims(img, axis=0)
        img = torch.as_tensor(img, dtype=torch.float32)
        img = img.to(device)
        outputs = model1(img)
        _, pred = torch.max(outputs, -1)

        if list(pred.detach().cpu().numpy())[0] == 2 or list(pred.detach().cpu().numpy())[0] == 5:
            outputs = model2(img)
            _, pred = torch.max(outputs, -1)
            if list(pred.detach().cpu().numpy())[0] == 0:
                results += [3]
            else:
                results += [6]
        else:
            results += [list(pred.detach().cpu().numpy())[0] + 1]

    print("--- inference is done! ---")
    print()
    df = pd.DataFrame(results)
    df.columns = ['prediction']
    df.to_csv(path, index=False)
    df.to_csv('/job_fair/submit.csv', index=False)
    print(df)
        
    
def read_json(path):
    with open(path) as json_file:
        json_data = json.load(json_file)
    return json_data
