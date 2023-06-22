from dataset.data import *
from network.head import *
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from util.meter import *
import time
from network.base_model import *
import random
from torchvision import datasets, transforms
import torch
import torch.backends.cudnn as cudnn
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='')
parser.add_argument('--port', type=int, default=23456)
parser.add_argument('--data_path', type=str, default='/mnt/data/dataset', help='path of dataset (default: \'./dataset\')')
parser.add_argument('--cos', type=str, default='cos', help='cosine decay mechanism for learning rate')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--gpuid', default='0', type=str, help='gpuid')
parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
parser.add_argument('--logdir', default='current', type=str, help='a part of checkpoint\'s name')
args = parser.parse_args()
# print(args)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def tsne_plot(save_dir, targets, outputs, epoch):
    print('generating t-SNE plot...')
    tsne = TSNE(random_state=epoch)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['classes'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='classes',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir, 'tsne'+str(epoch)+'.png'), bbox_inches='tight', dpi=1000)
    plt.clf()
    print('done!')

def main():
    setup_seed(1337)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    args.checkpoint = f'{args.name}-{args.dataset}-{args.logdir}.pth'

    print(args)
    if args.cos == 'cos':
        lr = 1
    else:
        lr = 10

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_path, download=True, transform=get_train_augment('cifar10'))
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=get_test_augment('cifar10'))
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data_path, download=True, transform=get_train_augment('cifar100'))
        test_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=get_test_augment('cifar100'))
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        train_dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200', train=True, transform=get_train_augment('tinyimagenet'))
        test_dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200', train=False, transform=get_test_augment('tinyimagenet'))
        num_classes = 200
    else:
        train_dataset = datasets.STL10(root=args.data_path, download=True, split='train', transform=get_train_augment('stl10'))
        test_dataset = datasets.STL10(root=args.data_path, download=True, split='test', transform=get_test_augment('stl10'))
        num_classes = 10

    pre_train = ModelBase(dataset=args.dataset)
    prefix = 'net.'

    state_dict = torch.load('./checkpoints/' + args.checkpoint, map_location='cpu')['model']
    # print(state_dict)
    for k in list(state_dict.keys()):
        if not k.startswith(prefix):
            del state_dict[k]
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict[k]
            del state_dict[k]
    pre_train.load_state_dict(state_dict)
    model = LinearHead(pre_train, dim_in=512, num_class=num_classes)
    model = model.cuda()
    # model = DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True)
    
    torch.backends.cudnn.benchmark = True

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    if args.cos == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * train_loader.__len__())
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    best_acc = 0
    best_acc5 = 0

    for epoch in range(args.epochs):
        # ---------------------- Train --------------------------
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            train_loader.__len__(),
            [losses, optimizer.param_groups[0]['lr']],
            prefix="Epoch: [{}]".format(epoch)
        )

        model.eval()
        for i, (image, label) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            out = model(image)
            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), image.size(0))

            # if i % 10 == 0:
            #     progress.display(i)

            if args.cos == 'cos':
                scheduler.step()
            
        if args.cos == 'step':
            scheduler.step()

        # ---------------------- Test --------------------------
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        # ---------------------- t-SNE --------------------------
        targets_list = list()
        outputs_list = list()
        # ---------------------- t-SNE --------------------------
        with torch.no_grad():
            # end = time.time()
            for i, (image, label) in enumerate(test_loader):
                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                # compute output
                output = model(image)

                outputs_np = output.data.cpu().numpy()
                targets_np = label.data.cpu().numpy()
                targets_list.append(targets_np[:, np.newaxis])
                outputs_list.append(outputs_np)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))

        sum1, cnt1, sum5, cnt5 = top1.sum, top1.count, top5.sum, top5.count

        top1_acc = sum1.float() / float(cnt1)
        top5_acc = sum5.float() / float(cnt5)

        best_acc = max(top1_acc, best_acc)
        best_acc5 = max(top5_acc, best_acc5)
        # if top1_acc > best_acc:
        #     best_acc = top1_acc
        #     if epoch==0 or epoch > 80:
        #         tsne_plot(args.save_dir, np.concatenate(targets_list, axis=0), np.concatenate(outputs_list, axis=0).astype(np.float64), epoch)
        print(
            'Epoch:{} * Acc@1 {top1_acc:.3f} Acc@5 {top5_acc:.3f} Best_Acc@1 {best_acc:.3f} Best_Acc@5 {best_acc5:.3f}'.format(
                epoch, top1_acc=top1_acc,
                top5_acc=top5_acc,
                best_acc=best_acc,
                best_acc5=best_acc5))
if __name__ == "__main__":
    main()
