import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import helper
from pathlib import Path
from tqdm import tqdm
import simulation
import torch
import torch.optim as optim

from torch.optim import lr_scheduler
import time
import copy
import random
from PIL import Image, ImageOps
from metrics import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchvision.datasets.folder import is_image_file, default_loader
import torch.utils.data as data
import dataset
from torchsummary import summary
import torch
import pytorch_unet
from utils import *
from pytorch_unet import UNetDS
from collections import defaultdict
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from loss import dice_loss
import argparse

from utils import load_weights, save_weights 

classes = ['Background', 'Sidelobe', 'Source', 'Galaxy']

metric_values = ['union', 'tp', 'fp', 'fn', 'obj_tp', 'obj_fp', 'obj_fn']
metric_names = ['accuracy', 'iou', 'precision', 'recall', 'dice', 'obj_precision', 'obj_recall']

DATA_PATH = Path('./data/')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", default='unet', type=str, choices=['no-skip', 'unet', 'unet-ds'], help="Weights path from which start training")
    parser.add_argument( "--resume", default='', type=str, help="Weights path from which start training")
    parser.add_argument( "--data_dir", default="data", help="Path of data folder")
    parser.add_argument( "--weights_dir", default ="weights", help="Weights dir where to save checkpoints")
    parser.add_argument( "--results_dir", default =".results", help="Weights dir where to store results")
    parser.add_argument( "--log_file", default ="log.txt", help="Log text file path")
    parser.add_argument( "--batch_size", type=int, default=20)
    parser.add_argument( "--lr", type=float, default=1e-4)
    parser.add_argument( "--epochs", type=int, default=100)
    parser.add_argument( "--n_classes", type=int, default=4)
    parser.add_argument( "--device", default="cuda")
    parser.add_argument( "--test", action="store_true")

    return parser.parse_args()

def main(args):

    # use same transform for train/val for this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    class JointRandomHorizontalFlip(object):
        """Randomly horizontally flips the given list of PIL.Image with a probability of 0.5
        """

        def __call__(self, imgs):
            if random.random() < 0.5:
                return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
            return imgs

    normalize = transforms.Normalize(mean=dataset.mean, std=dataset.std)
    train_joint_transformer = transforms.Compose([
        #joint_transforms.JointRandomCrop(224), # commented for fine-tuning
        JointRandomHorizontalFlip()
        ])
    train_dset = dataset.RGDataset(DATA_PATH, 'train',
        joint_transform=train_joint_transformer,
        transform=transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=transforms.Compose([
            transforms.Resize([224, 224]),
            dataset.LabelToLongTensor(),
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=args.batch_size, num_workers=16, persistent_workers=True, drop_last=True, shuffle=True)

    val_dset = dataset.RGDataset(
        DATA_PATH, 'val', joint_transform=None,
        transform=transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize
        ]),
        target_transform=transforms.Compose([
            transforms.Resize([224, 224]),
            dataset.LabelToLongTensor(),
        ]))
        
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False)


    dataloaders = {
        'train': train_loader,
        'val': val_loader,
    }

    num_class = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == 'no-skip':
        model = pytorch_unet.UNetNoSkip(num_class).to(device)
    elif args.model == 'unet':
        model = pytorch_unet.UNetSkip(num_class).to(device)
    elif args.model == 'unet-ds':
        model = pytorch_unet.UNetDS(num_class).to(device)

    summary(model, input_size=(3, 224, 224))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

    if args.resume:
        load_weights(model, args.resume)

    # train_model(model, optimizer_ft, dataloaders, n_samples, args, exp_lr_scheduler)
    best_model_wts = copy.deepcopy(model.state_dict())
    

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        
        since = time.time()

        trn_loss, trn_metrics, = train(model, optimizer_ft, dataloaders['train'], args, exp_lr_scheduler)
        print_detection_metrics(epoch, trn_loss, trn_metrics, 'train')

        val_loss, best_loss, val_metrics = validate(model, dataloaders['val'], epoch, args)
        print_detection_metrics(epoch, val_loss, val_metrics, 'val')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val loss: {:4f}'.format(best_loss))


def train(model, optimizer, loader, args, scheduler=None):

    losses = []

    model.train()  # Set model to training mode

    metrics = defaultdict(float)
    trn_metrics = {class_name: {metric_name: 0. for metric_name in metric_names} for class_name in classes[1:]}
    batch_metrics = {class_name: {metric_name: [] for metric_name in metric_names} for class_name in classes[1:]}
    
    for inputs, labels in tqdm(loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)             

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        if(isinstance(model, UNetDS)):
            out, up_outs = model(inputs)
            outputs = resize(out, (132,132))
            labels = resize(labels.float(), (132,132))
            loss = calc_loss(outputs, labels, metrics)
            for up_out in up_outs:
                up_out = resize(up_out, (132,132))
                loss += calc_loss(up_out, labels, metrics)
        else:
            outputs = resize(model(inputs), (132,132))
            labels = resize(labels.float(), (132,132))
            loss = calc_loss(outputs, labels, metrics)

        loss.backward()
        optimizer.step()

        losses.append(loss.detach().item())
        preds = get_predictions(outputs.detach())
        labels = get_predictions(labels)

    trn_loss = np.mean(losses)
    for class_name in classes[1:]:
        trn_metrics[class_name] = {metric_name: np.mean(batch_metrics[class_name][metric_name]) for metric_name in metric_names}

    return trn_loss, trn_metrics
    
def validate(model, loader, epoch, args):
    
    losses = []
    best_loss = 1e10
    model.eval()   # Set model to evaluate mode

    metrics = defaultdict(float)
    metrics = {class_name: {metric_name: 0. for metric_name in metric_names} for class_name in classes}
    batch_metrics = {class_name: {metric_name: [] for metric_name in metric_names} for class_name in classes}
    epoch_samples = 0
    
    for inputs, labels in tqdm(loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)             

        # forward
        # track history if only in train
        if(isinstance(model, UNetDS)):
            out, up_outs = model(inputs)
            outputs = resize(out, (132,132))
            labels = resize(labels.float(), (132,132))
            loss = calc_loss(outputs, labels, metrics)
            for up_out in up_outs:
                up_out = resize(up_out, (132,132))
                loss += calc_loss(up_out, labels, metrics)
        else:
            outputs = resize(model(inputs), (132,132))
            labels = resize(labels.float(), (132,132))
            loss = calc_loss(outputs, labels, metrics)

        losses.append(loss.item())
        preds = get_predictions(outputs.detach())
        labels = get_predictions(labels)

    epoch_loss = np.mean(losses)
    for class_name in classes[1:]:
            metrics[class_name] = {metric_name: np.mean(batch_metrics[class_name][metric_name]) for metric_name in metric_names}
    
    # deep copy the model
    if epoch_loss < best_loss:
        print("saving best model")
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        save_weights(model, args.model, epoch, epoch_loss, 0.8)

    return epoch_loss, best_loss, metrics


if __name__ == '__main__':

    args = get_args()

    main(args)