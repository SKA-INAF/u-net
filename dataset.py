import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import is_image_file, default_loader
import torchvision.transforms as transforms

classes = ['Background', 'Sidelobe', 'Source', 'Galaxy']

# https://github.com/yandex/segnet-torch/blob/master/datasets/camvid-gen.lua
# class_weight = torch.FloatTensor([
#     0.58872014284134, 0.51052379608154, 2.6966278553009,
#     0.45021694898605, 1.1785038709641, 0.77028578519821, 2.4782588481903,
#     2.5273461341858, 1.0122526884079, 3.2375309467316, 4.1312313079834, 0])

# class_weight = torch.FloatTensor([
#     0.1, 0.1, 0.1,
#     0.1, 0.1, 0.1, 0.1,
#     0.1, 2.009, 0.411, 3.80, 0])

#class_weight = torch.FloatTensor([0.25, 4.387, 0.3107, 1.807])
class_weight = torch.FloatTensor([0.25, 2.85, 0.30, 1.50])

mean = [0.28535324, 0.28535324, 0.28535324]
std = [0.28536762, 0.28536762, 0.28536762]
#mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
#std = [0.27413549931506, 0.28506257482912, 0.28284674400252]



class_color = [
    (0, 0, 0),    
    (32,207,227),
    (250,7,7),
    (237,237,12),
]


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


class LabelToLongTensor(object):
    def __call__(self, pic):        
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
        return label


class LabelTensorToPILImage(object):
    def __call__(self, label):
        label = label.unsqueeze(0)
        colored_label = torch.zeros(3, label.size(1), label.size(2)).byte()
        for i, color in enumerate(class_color):
            mask = label.eq(i)
            for j in range(3):
                colored_label[j].masked_fill_(mask, color[j])
        npimg = colored_label.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        mode = None
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]
            mode = "L"

        return Image.fromarray(npimg, mode=mode)


class RGDataset(data.Dataset):

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=None,
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.class_weight = class_weight
        self.classes = classes
        self.mean = mean
        self.std = std

        if download:
            self.download()

        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        num_classes = 4
        target = Image.open(path.replace(self.split, self.split + 'annot'))

        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])

        if self.transform is not None:
            img = self.transform(img)
        
        target = self.target_transform(target)
        masks = []
        for i in range(num_classes):
            masks.append(torch.where(target == i, 1, 0).unsqueeze(0))
        target = torch.cat(masks)
        return img, target

    def __len__(self):
        return len(self.imgs)


class TestRGDataset(data.Dataset):

    def __init__(self, root, split='test', joint_transform=None,
                 transform=None, target_transform=None,
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.class_weight = class_weight
        self.classes = classes
        self.mean = mean
        self.std = std

        if download:
            self.download()

        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        num_classes = 4
        target = Image.open(path.replace(self.split, self.split + 'annot'))

        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])

        if self.transform is not None:
            img = self.transform(img)
        
        target = self.target_transform(target)
        masks = []
        for i in range(num_classes):
            masks.append(torch.where(target == i, 1, 0).unsqueeze(0))
        target = torch.cat(masks)
        return img, target, path

    def __len__(self):
        return len(self.imgs)