import numpy as np
from pathlib import Path
import torch
from scipy import ndimage

classes = ['Background', 'Sidelobe', 'Source', 'Galaxy']

# metric_values = ['union', 'tp', 'fp', 'tn', 'fn', 'obj_tp', 'obj_fp', 'obj_fn']
# metrics = ['accuracy', 'iou', 'precision', 'sensitivity', 'specificity', 'dice', 'obj_precision', 'obj_recall']

def compute_union(preds, targets, class_id):
    total_union = {}
    current_class = torch.where(preds == class_id, 1.,0.) # isolates the class of interest
    gt = torch.where(targets == class_id, 1., 0.)
    union = torch.where(torch.logical_or(current_class, gt), 1., 0.)

    total_union = union.sum().item()
    
    return total_union
    
def compute_batch_metrics(union, tp, fp, fn, tn):

    # TODO IoU and Dice are the same metric, remove?

    accuracy       =   division(tp + tn, tp + fp + tn + fn)
    iou            =   division(tp, union)
    precision      =   division(tp, tp + fp)
    recall         =   division(tp, tp + fn)
    dice           =   division(tp, tp + fp + fn)

    return accuracy, iou, precision, recall, dice

def compute_batch_obj_metrics(obj_tp, obj_fp, obj_fn):

    obj_precision  =   division(obj_tp, obj_tp + obj_fp)
    obj_recall     =   division(obj_tp, obj_tp + obj_fn)

    return obj_precision, obj_recall

def compute_object_confusion_matrix(preds, targets, class_id, threshold=0.5):

    tp = 0
    fp = 0
    fn = 0

    for pred, target in zip(preds, targets):

        gt = torch.where(target == class_id, 1., 0.)
        current_class = torch.where(pred == class_id, 1., 0.) # isolates the class of interest
        pred_objects, nr_pred_objects = ndimage.label(current_class.cpu())
        target_objects, nr_target_objects = ndimage.label(gt.cpu())
        pred_objects = torch.from_numpy(pred_objects).to(pred.device)
        target_objects = torch.from_numpy(target_objects).to(pred.device)

        for pred_idx in range(nr_pred_objects):
            current_obj_pred = torch.where(pred_objects == pred_idx, 1., 0.)

            obj_iou = get_obj_iou(nr_target_objects, target_objects, current_obj_pred)
            if nr_target_objects != 0:
                if obj_iou >= threshold:
                    tp += 1
                else: 
                    fp += 1

        if nr_target_objects > nr_pred_objects:
            fn += (nr_target_objects - nr_pred_objects)
    
    return tp, fp, fn

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


def convert_preds(preds_json, preds, paths):

    tp = 0
    fp = 0
    fn = 0
    for pred, path in zip(preds, paths):
        img_preds = {'labels': [], 'boxes': [], 'scores': []}

        img_name = Path(path).stem

        for i, class_name in enumerate(classes[1:]):
            current_class = torch.where(pred == i, 1., 0.) # isolates the class of interest
            pred_objects, nr_pred_objects = ndimage.label(current_class.cpu())
            pred_objects = torch.from_numpy(pred_objects).to(pred.device)
        

            for pred_idx in range(nr_pred_objects):
                current_obj_pred = torch.where(pred_objects == pred_idx + 1, 1., 0.)
                box = masks_to_boxes(current_obj_pred.unsqueeze(0))
                img_preds['boxes'].append(box.tolist())
                img_preds['labels'].append(class_name)
                img_preds['scores'].append(0.9) # Fake score just to adapt to metric format

        preds_json[img_name] = img_preds


def get_obj_iou(nr_target_objects, target_objects, current_obj_pred):
    obj_ious = []
    for target_idx in range(nr_target_objects):
        current_obj_target = target_objects == target_idx
        intersection = torch.where(torch.logical_and(current_obj_pred, current_obj_target), 1., 0.)
        union = torch.where(torch.logical_or(current_obj_pred, current_obj_target), 1., 0.)

        obj_ious.append(intersection.sum().item() / union.sum().item())
    if len(obj_ious) > 0:
        return np.nanmax(obj_ious).item()
    else:
        return 0 

def compute_confusion_matrix(preds, targets, class_id):

    assert preds.size() == targets.size()
    current_class = preds == class_id # isolates the class of interest
    gt = targets == class_id

    tp = gt.mul(current_class).eq(1).sum().item()
    fp = gt.eq(0).long().mul(current_class).eq(1).sum().item()
    fn = current_class.eq(0).long().mul(gt).eq(1).sum().item()
    tn = current_class.eq(0).long().mul(gt).eq(0).sum().item()

    return tp, fp, fn, tn

def division(x,y):
    return x / y if y else 0

def get_count(tensor):
    for i in range(4):
        print(f'{str(i)}: {(tensor == i).sum()}')



def compute_final_metrics(metrics, eps=1e-6):
    final_metrics = {}

    final_metrics['accuracy']       =   division(metrics['tp'], (metrics['tp'] + metrics['fn']))
    final_metrics['iou']            =   division(metrics['tp'], metrics['union'])
    final_metrics['recall']         =   division(metrics['tp'], (metrics['tp'] + metrics['fn']))
    final_metrics['precision']      =   division(metrics['tp'], (metrics['tp'] + metrics['fp']))
    final_metrics['dice']           =   division(metrics['tp'], (metrics['tp'] + metrics['fp'] + metrics['fn']))
    final_metrics['obj_precision']  =   division(metrics['obj_tp'], (metrics['obj_tp'] + metrics['obj_fp']))
    final_metrics['obj_recall']     =   division(metrics['obj_tp'], (metrics['obj_tp'] + metrics['obj_fn']))

    return final_metrics



def print_detection_metrics(epoch, loss, trn_metrics, phase):
    #print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, trn_loss, trn_acc))    
    print('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, loss))
    print('Per class metrics: ')
    for i, class_name in enumerate(classes[1:]):
        if trn_metrics[class_name] != {}:
            print(f'\t {class_name}: \tAcc: {trn_metrics[class_name]["accuracy"]:.4f}, \tIoU: {trn_metrics[class_name]["iou"]:.4f}, \
                 \tPrecision: {trn_metrics[class_name]["precision"]:.4f}, \tRecall: {trn_metrics[class_name]["recall"]:.4f},  \
                 \n\tObject Precision: {trn_metrics[class_name]["obj_precision"]:.4f}, \tObject Recall: {trn_metrics[class_name]["obj_recall"]:.4f}, \
                 \n\tF1-Score: {trn_metrics[class_name]["f1-score"]:.4f}, \tObject F1-Score: {trn_metrics[class_name]["obj_f1-score"]:.4f}')

    with open('out.txt', 'a') as out:
        out.write('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, loss)+ '\n')
        out.write('Per class metrics: \n')
        for i, class_name in enumerate(classes[1:]):
            if trn_metrics[class_name] != {}:
                out.write(f'\t {class_name}: \tAcc: {trn_metrics[class_name]["accuracy"]:.4f}, \tIoU: {trn_metrics[class_name]["iou"]:.4f}, \
                 \tPrecision: {trn_metrics[class_name]["precision"]:.4f}, \tRecall: {trn_metrics[class_name]["recall"]:.4f},  \
                 \n\tObject Precision: {trn_metrics[class_name]["obj_precision"]:.4f}, \tObject Recall: {trn_metrics[class_name]["obj_recall"]:.4f}, \
                 \n\tF1-Score: {trn_metrics[class_name]["f1-score"]:.4f}, \tObject F1-Score: {trn_metrics[class_name]["obj_f1-score"]:.4f}')
