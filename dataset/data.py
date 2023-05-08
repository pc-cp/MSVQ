from torchvision import datasets, transforms
from PIL import ImageFilter, Image
import numpy as np
import os
import sys
import random
from torch.utils.data import Dataset

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


class ThreeCrop:
    def __init__(self, strong, weak):
        self.strong = strong
        self.weak = weak

    def __call__(self, img):
        im_1 = self.strong(img)
        im_2 = self.weak(img)
        im_3 = self.weak(img)
        im_4 = self.weak(img)

        return im_1, im_2, im_3, im_4

class STL10Pair(datasets.STL10):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)

        self.weak_aug = weak_aug

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            
            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
                pos_3 = self.weak_aug(img)
                pos_4 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)
                pos_3 = self.transform(img)
                pos_4 = self.transform(img)
        return pos_1, pos_2, pos_3, pos_4

class CIFAR10Pair(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.weak_aug = weak_aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
                pos_3 = self.weak_aug(img)
                pos_4 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)
                pos_3 = self.transform(img)
                pos_4 = self.transform(img)
        return pos_1, pos_2, pos_3, pos_4

class CIFAR100Pair(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.weak_aug = weak_aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
                pos_3 = self.weak_aug(img)
                pos_4 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)
                pos_3 = self.transform(img)
                pos_4 = self.transform(img)
        return pos_1, pos_2, pos_3, pos_4


# pretrain-strong aug
def get_contrastive_augment(dataset):
    size = 32
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        
    normalize = transforms.Normalize(mean=mean, std=std)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform
# pretrain-weak aug
def get_weak_augment(dataset):
    size = 32
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(0.9),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform
# downstream work fine-tune
def get_train_augment(dataset):
    size = 32
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        
    normalize = transforms.Normalize(mean=mean, std=std)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform
# downstream work evaluation
def get_test_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean=mean, std=std)
    
    # if dataset == 'stl10':
    #     test_transform = transforms.Compose([
    #         transforms.Resize(70),
    #         transforms.CenterCrop(64),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # else:
    test_transform = transforms.Compose([
        # transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform
