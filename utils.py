import torch
import shutil
import os
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