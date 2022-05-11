import numpy as np
import torch
from utils import *
from pathlib import Path
from tqdm import tqdm
import torch

import time
from metrics import *
from torch.utils.data import DataLoader
import json
from torchvision import transforms
import dataset
from torchsummary import summary
import torch
import pytorch_unet
from pytorch_unet import UNetDS
from collections import defaultdict
from torchvision.transforms.functional import resize
import argparse

from utils import load_weights, save_weights 

classes = ['Background', 'Galaxy', 'Source', 'Sidelobe']

metric_values = ['union', 'tp', 'fp', 'fn', 'obj_tp', 'obj_fp', 'obj_fn']
metric_names = ['accuracy', 'iou', 'precision', 'recall', 'dice', 'obj_precision', 'obj_recall']

DATA_PATH = Path('./data/')

def get_args():
    # TODO Model selection
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

    normalize = transforms.Normalize(mean=dataset.mean, std=dataset.std)


    test_dset = dataset.TestRGDataset(
        args.data_dir, 'test', joint_transform=None,
        transform=transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize
        ]),
        target_transform=transforms.Compose([
            transforms.Resize([224, 224]),
            dataset.LabelToLongTensor(),
        ]))
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False)

    num_class = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == 'no-skip':
        model = pytorch_unet.UNetNoSkip(num_class).to(device)
    elif args.model == 'unet':
        model = pytorch_unet.UNetSkip(num_class).to(device)
    elif args.model == 'unet-ds':
        model = pytorch_unet.UNetDS(num_class).to(device)

    summary(model, input_size=(3, 224, 224))

    assert args.resume, 'Required path for loading weights'
    load_weights(model, args.resume)

    print('Testing model and computing metrics')
    print('-' * 10)
    
    since = time.time()

    test_loss, test_metrics = test(model, test_loader, args)
    print_detection_metrics(0, test_loss, test_metrics, 'test')

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))




    
def test(model, loader, args):
    
    losses = []
    preds_json = {}
    gt_json = {}
    model.eval()   # Set model to evaluate mode

    metrics = defaultdict(float)
    metrics = {class_name: {metric_name: 0. for metric_name in metric_names} for class_name in classes}
    batch_metrics = {class_name: {metric_name: [] for metric_name in metric_names} for class_name in classes}
    epoch_samples = 0
    
    for inputs, labels, paths in tqdm(loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)             

        # forward
        # track history if only in train
        if(isinstance(model, UNetDS)):
            with torch.no_grad():
                out, up_outs = model(inputs)
            outputs = resize(out, (132,132))
            labels = resize(labels.float(), (132,132))
            loss = calc_loss(outputs, labels, metrics)
            for up_out in up_outs:
                up_out = resize(up_out, (132,132))
                loss += calc_loss(up_out, labels, metrics)
        else:
            with torch.no_grad():
                outputs = resize(model(inputs), (132,132))
            labels = resize(labels.float(), (132,132))
            loss = calc_loss(outputs, labels, metrics)

        losses.append(loss.item())
        preds = get_predictions(outputs.detach())
        labels = get_predictions(labels)
        for i, class_name in enumerate(classes[1:]):
            union = compute_union(preds, labels, i + 1) 
            
            if union == 0:
                # There is no object with that class, skipping...
                continue

            tp, fp, fn, tn = compute_confusion_matrix(preds, labels, i + 1)
            obj_tp, obj_fp, obj_fn = compute_object_confusion_matrix(preds, labels, i + 1, 0.9)

            accuracy, iou, precision, recall, dice = compute_batch_metrics(union, tp, fp, fn, tn)
            obj_precision, obj_recall = compute_batch_obj_metrics(obj_tp, obj_fp, obj_fn)

            batch_metrics[class_name]['accuracy'].append(accuracy)
            batch_metrics[class_name]['iou'].append(iou)
            batch_metrics[class_name]['precision'].append(precision)
            batch_metrics[class_name]['recall'].append(recall)
            batch_metrics[class_name]['dice'].append(dice)
            batch_metrics[class_name]['obj_precision'].append(obj_precision)
            batch_metrics[class_name]['obj_recall'].append(obj_recall)

        # convert_preds(preds_json, preds, paths)
        # convert_preds(gt_json, labels, paths)


    epoch_loss = np.mean(losses)
    for class_name in classes[1:]:
            metrics[class_name] = {metric_name: np.mean(batch_metrics[class_name][metric_name]) for metric_name in metric_names}
            metrics[class_name]['f1-score'] = \
                (2 * metrics[class_name]['precision'] * metrics[class_name]['recall']) / \
                    (metrics[class_name]['precision'] + metrics[class_name]['recall'])
            metrics[class_name]['obj_f1-score'] = \
                (2 * metrics[class_name]['obj_precision'] * metrics[class_name]['obj_recall']) / \
                    (metrics[class_name]['obj_precision'] + metrics[class_name]['obj_recall'])

    # with open(f'output_json/{args.model}/preds.json', 'w') as pj:
    #     json.dump(preds_json, pj)
    # with open(f'output_json/{args.model}/gt.json', 'w') as gtj:
    #     json.dump(gt_json, gtj)

    return epoch_loss, metrics


if __name__ == '__main__':

    args = get_args()

    main(args)