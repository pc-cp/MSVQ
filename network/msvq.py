from network.base_model import ModelBase_ResNet18
from network.head import MoCoHead
import torch.nn.functional as F
import copy
from util.MemoryBankModule import MemoryBankModule
from util.utils import *

class MSVQ(nn.Module):
    def __init__(self, dim=128, K=4096, m1=0.99, m2=0.95, tem=0.05, dataset='cifar10', bn_splits=8, symmetric=False):
        super(MSVQ, self).__init__()
        self.K = K
        self.m1 = m1
        self.m2 = m2
        self.tem = tem
        self.symmetric = symmetric
        # create the encoders, [net == f_s]
        self.net  = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.f_t1 = copy.deepcopy(self.net)
        self.f_t2 = copy.deepcopy(self.net)

        self.g_s  = MoCoHead(input_dim=512, out_dim=dim)
        self.g_t1 = copy.deepcopy(self.g_s)
        self.g_t2 = copy.deepcopy(self.g_s)

        self.queue1 = MemoryBankModule(size=self.K).cuda()
        self.queue2 = MemoryBankModule(size=self.K).cuda()

        deactivate_requires_grad(self.f_t1)
        deactivate_requires_grad(self.f_t2)
        deactivate_requires_grad(self.g_t1)
        deactivate_requires_grad(self.g_t2)
    def contrastive_loss(self, im_1, im_2, im_3, im_4, labels, update=False):

        # compute query features
        z_1 = self.g_s(self.net(im_1))  # queries: NxC

        with torch.no_grad():  # no gradient to keys
            # shuffle
            im_2_, shuffle = batch_shuffle(im_2)
            z_2 = self.g_t1(self.f_t1(im_2_))  # keys: NxC
            # undo shuffle
            z_2 = batch_unshuffle(z_2, shuffle)

            # shuffle
            im_3_, shuffle = batch_shuffle(im_3)
            z_3 = self.g_t1(self.f_t1(im_3_))  # keys: NxC
            # undo shuffle
            z_3 = batch_unshuffle(z_3, shuffle)

            # shuffle
            im_4_, shuffle = batch_shuffle(im_4)
            z_4 = self.g_t2(self.f_t2(im_4_))  # keys: NxC
            # undo shuffle
            z_4 = batch_unshuffle(z_4, shuffle)

        # Nearest Neighbour,    queue: [feature_dim, self.K]
        _, queue_1, _ = self.queue1(output=z_2, labels=labels, update=update)
        _, queue_2, _ = self.queue1(output=z_4, labels=labels, update=update)
        # ================normalized==================
        z_1 = nn.functional.normalize(z_1, dim=1)
        z_2, z_3, z_4 = nn.functional.normalize(z_2, dim=1), nn.functional.normalize(z_3, dim=1), nn.functional.normalize(z_4, dim=1)

        queue_1, queue_2 = queue_1.t(), queue_2.t()
        queue_1, queue_2 = nn.functional.normalize(queue_1, dim=1), nn.functional.normalize(queue_2, dim=1)

        # ===========MSVQ=============
        # calculate similiarities, logits_q_queue has shape (n, self.K), logits_z_k_queue has shape (n, self.K)
        logits_11 = torch.einsum("nc,mc->nm", z_1, queue_1)
        logits_12 = torch.einsum("nc,mc->nm", z_1, queue_2)

        logits_21 = torch.einsum("nc,mc->nm", z_2, queue_1)
        logits_31 = torch.einsum("nc,mc->nm", z_3, queue_1)
        logits_42 = torch.einsum("nc,mc->nm", z_4, queue_2)

        loss1 = - torch.sum(F.softmax(logits_21.detach() / self.tem, dim=1) * F.log_softmax(logits_11 / torch.tensor(0.1), dim=1), dim=1).mean().cuda()
        loss2 = - torch.sum(F.softmax(logits_31.detach() / self.tem, dim=1) * F.log_softmax(logits_11 / torch.tensor(0.1), dim=1), dim=1).mean().cuda()
        loss3 = - torch.sum(F.softmax(logits_42.detach() / self.tem, dim=1) * F.log_softmax(logits_12 / torch.tensor(0.1), dim=1), dim=1).mean().cuda()
        loss = (loss1+loss2+loss3)/3.0

        return loss
    def forward(self, im_1, im_2, im_3, im_4, labels):
        # Updates parameters of `model_ema` with Exponential Moving Average of `model`
        update_momentum(model=self.net, model_ema=self.f_t1, m=self.m1)
        update_momentum(model=self.net, model_ema=self.f_t2, m=self.m2)

        update_momentum(model=self.g_s, model_ema=self.g_t1, m=self.m1)
        update_momentum(model=self.g_s, model_ema=self.g_t2, m=self.m2)

        loss_12 = self.contrastive_loss(im_1, im_2, im_3, im_4, update=True, labels=labels)

        loss = loss_12

        return loss
