import torch
from util.meter import *
from network.MSVQ import MSVQ
import time
from dataset.data import *
import math
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
# import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='', help='Specify the method name(msvq)')
parser.add_argument('--doc', type=str, default='Test', help='To describe what this training is about')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--data_path', type=str, default='/mnt/data/dataset', help='path of dataset')
parser.add_argument('--port', type=int, default=23456)

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--queue_size', type=int, default=4096, help='Queue size')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--base_lr', type=float, default=0.06)
parser.add_argument('--tem', type=float, default=0.1, help='Temperature used in the loss function')
parser.add_argument('--m1', type=float, default=0.95, help='momentum for teacher1')
parser.add_argument('--m2', type=float, default=0.99, help='momentum for teacher2')

parser.add_argument('--weak', default=False, action='store_true', help='weak aug for teacher')
parser.add_argument('--symmetric', default=False, action='store_true', help='use a symmetric loss function that backprops to both crops')

parser.add_argument('--gpuid', default='1', type=str, help='gpuid')
parser.add_argument('--logdir', default='current', type=str, help='log')
args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch, i, iteration_per_epoch, args):
    warm_up = 5
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (args.epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = args.base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * args.base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def train(train_loader, model, optimizer, epoch, iteration_per_epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    ce_losses = AverageMeter('CE', ':.4e')
    purity_ave = AverageMeter('PUR', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, ce_losses, purity_ave, optimizer.param_groups[0]['lr']],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    for i, ((im_1, im_2, im_3, im_4), labels) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, iteration_per_epoch, args)
        data_time.update(time.time() - end)

        im_1 = im_1.cuda(non_blocking=True)
        im_2 = im_2.cuda(non_blocking=True)
        im_3 = im_3.cuda(non_blocking=True)
        im_4 = im_4.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        loss = model(im_1, im_2, im_3, im_4, labels=labels)

        # record loss
        ce_losses.update(loss.item(), im_1.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure time
        batch_time.update(time.time() - end)
        end = time.time()
        # progress.display(i)
    return ce_losses.avg

# test using a knn monitor
def online_test(net, memory_data_loader, test_data_loader, args):
    net.eval()
    classes = args.num_classes
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for i, (data, target) in enumerate(memory_data_loader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]: D represents the number of feature dimensions of each image, N represents the size of dataset
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.cat(target_bank, dim=0).t().contiguous()
        # [N]
        for i, (data, target) in enumerate(test_data_loader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            feature = net(data)
            feature = F.normalize(feature, dim=1)
            # same with moco
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    return total_top1 / total_num * 100

def main():
    # args.name = 'msvq'
    # args.logdir = 'cifar10_00'
    # setup_seed(1337)
    # args.gpuid = '1'
    # args.tem = 0.04

    # args.weak = True
    # args.m1 = 0.99
    # args.m2 = 0.95
    print(args)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    if args.name == 'msvq':
        model = MSVQ(K=args.queue_size, m1=args.m1, m2=args.m2, tem=args.tem, dataset=args.dataset, symmetric=args.symmetric)
    else:
        print(' Sorry, this repository is containing MSVQ. ')

    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
    torch.backends.cudnn.benchmark = True
    
    if args.dataset == 'cifar10':
        if args.weak:
            dataset = CIFAR10Pair(root=args.data_path, download=True, transform=get_contrastive_augment('cifar10'), weak_aug=get_weak_augment('cifar10'))
        else:
            dataset = CIFAR10Pair(root=args.data_path, download=True, transform=get_contrastive_augment('cifar10'), weak_aug=None)
        memory_dataset = datasets.CIFAR10(root=args.data_path, download=True, transform=get_test_augment('cifar10'))
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=get_test_augment('cifar10'))
        args.num_classes = 10
    elif args.dataset == 'stl10':
        if args.weak:
            dataset = STL10Pair(root=args.data_path, download=True, split='train+unlabeled',
                                transform=get_contrastive_augment('stl10'), weak_aug=get_weak_augment('stl10'))
        else:
            dataset = STL10Pair(root=args.data_path, download=True, split='train+unlabeled', transform=get_contrastive_augment('stl10'), weak_aug=None)
        memory_dataset = datasets.STL10(root=args.data_path, download=True, split='train', transform=get_test_augment('stl10'))
        test_dataset = datasets.STL10(root=args.data_path, download=True, split='test', transform=get_test_augment('stl10'))
        args.num_classes = 10
    elif args.dataset == 'tinyimagenet':
        if args.weak:
            dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200', train=True,
                                   transform=FourCrop(get_contrastive_augment('tinyimagenet'),
                                                       get_weak_augment('tinyimagenet')))
        else:
            dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200', train=True, transform=FourCrop(get_contrastive_augment('tinyimagenet'), get_contrastive_augment('tinyimagenet')))
        memory_dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200', train=True, transform=get_test_augment('tinyimagenet'))
        test_dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200', train=False, transform=get_test_augment('tinyimagenet'))
        args.num_classes = 200
    else:
        if args.weak:
            dataset = CIFAR100Pair(root=args.data_path, download=True, transform=get_contrastive_augment('cifar100'),
                                   weak_aug=get_weak_augment('cifar100'))
        else:
            dataset = CIFAR100Pair(root=args.data_path, download=True, transform=get_contrastive_augment('cifar100'), weak_aug=None)
        memory_dataset = datasets.CIFAR100(root=args.data_path, download=True, transform=get_test_augment('cifar100'))
        test_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=get_test_augment('cifar100'))
        args.num_classes = 100
    
    train_loader = DataLoader(dataset, shuffle=True, num_workers=6, pin_memory=True, batch_size=args.batch_size, drop_last=True)
    memory_loader = DataLoader(memory_dataset, shuffle=False, num_workers=6, pin_memory=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=6, pin_memory=True, batch_size=args.batch_size)
    iteration_per_epoch = train_loader.__len__()
    
    checkpoint_path = 'checkpoints/'+args.name+'-{}-{}.pth'.format(args.dataset, args.logdir)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(checkpoint_path, 'found, start from epoch', start_epoch)
    else:
        start_epoch = 0
        print(checkpoint_path, 'not found, start from epoch 0')

    model.train()
    best_acc = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(train_loader, model, optimizer, epoch, iteration_per_epoch, args)
        cur_acc = online_test(model.net, memory_loader, test_loader, args)
        if cur_acc > best_acc:
            best_acc = cur_acc

        print(f'Epoch [{epoch}/{args.epochs}]: 200-NN-Best: {best_acc:.4f}!, 200-NN: {cur_acc:.4f}, loss: {train_loss:.4f}')
        if epoch == args.epochs-1:
            torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)

if __name__ == "__main__":
    main()