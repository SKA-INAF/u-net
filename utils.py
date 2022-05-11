import torch
import shutil
import os
import torch.nn.functional as F
from loss import dice_loss
WEIGHTS_PATH = 'weights'


def load_weights(model, fpath, device="cuda"):
    print("loading weights '{}'".format(fpath))
    if device == 'cpu':
        weights = torch.load(fpath, map_location=torch.device('cpu'))
    else:
        weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'], strict=False)
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['accuracy']))
    return startEpoch

def save_weights(model, model_name, epoch, loss, acc): #err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, acc)
    weights_fpath = os.path.join(WEIGHTS_PATH, model_name, weights_fname)
    os.makedirs(os.path.join(WEIGHTS_PATH, model_name), exist_ok=True)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            #'error': err,
            'accuracy': acc,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH + '/latest.pth')


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))  


def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.max(1)
    indices = indices.view(bs,h,w)
    return indices