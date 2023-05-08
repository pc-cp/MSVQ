import numpy as np
import torch.nn as nn
import torch
from network.base_model import ModelBaseMoCo
from network.head import MoCoHead
import torch.nn.functional as F

class MSVQ(nn.Module):
    def __init__(self, dim=128, K=4096, mk=0.99, mp=0.95, tem=0.04, dataset='cifar10', bn_splits=8):
        super(MSVQ, self).__init__()

        self.K = K
        self.mk = mk
        self.mp = mp
        self.tem = tem
        # create the encoders
        self.net       = ModelBaseMoCo(dataset=dataset, bn_splits=bn_splits)
        self.encoder_k = ModelBaseMoCo(dataset=dataset, bn_splits=bn_splits)
        self.encoder_p = ModelBaseMoCo(dataset=dataset, bn_splits=bn_splits)

        self.head_q = MoCoHead(input_dim=512)
        self.head_k = MoCoHead(input_dim=512)
        self.head_p = MoCoHead(input_dim=512)

        for param_q, param_k, param_p in zip(self.net.parameters(), self.encoder_k.parameters(), self.encoder_p.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

            param_p.data.copy_(param_q.data)  # initialize
            param_p.requires_grad = False  # not update by gradient

        for param_q, param_k, param_p in zip(self.head_q.parameters(), self.head_k.parameters(), self.head_p.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

            param_p.data.copy_(param_q.data)  # initialize
            param_p.requires_grad = False  # not update by gradient

        # self.max_entropy = np.log(self.K)

        # create the queue
        self.register_buffer("queue_k", torch.randn(dim, self.K))
        self.queue_k = nn.functional.normalize(self.queue_k, dim=0)

        self.register_buffer("queue_ptr_k", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_p", torch.randn(dim, self.K))
        self.queue_p = nn.functional.normalize(self.queue_p, dim=0)

        self.register_buffer("queue_ptr_p", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k, param_p in zip(self.net.parameters(), self.encoder_k.parameters(), self.encoder_p.parameters()):
            param_k.data = param_k.data * self.mk + param_q.data * (1. - self.mk)
            param_p.data = param_p.data * self.mp + param_q.data * (1. - self.mp)

        for param_q, param_k, param_p in zip(self.head_q.parameters(), self.head_k.parameters(), self.head_p.parameters()):
            param_k.data = param_k.data * self.mk + param_q.data * (1. - self.mk)
            param_p.data = param_p.data * self.mp + param_q.data * (1. - self.mp)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, p):
        batch_size = keys.shape[0]

        ptr_k = int(self.queue_ptr_k)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_k[:, ptr_k:ptr_k + batch_size] = keys.t()  # transpose
        ptr_k = (ptr_k + batch_size) % self.K  # move pointer

        self.queue_ptr_k[0] = ptr_k

        # batch_size = keys.shape[0]

        ptr_p = int(self.queue_ptr_p)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_p[:, ptr_p:ptr_p + batch_size] = p.t()  # transpose
        ptr_p = (ptr_p + batch_size) % self.K  # move pointer

        self.queue_ptr_p[0] = ptr_p

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def forward(self, im1, im2, im3, im4):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """        

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        
        # compute query features
        q = self.net(im1)  # queries: NxC
        q = self.head_q(q)
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im2)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = self.head_k(k)
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

            im_p_, idx_unshufflek = self._batch_shuffle_single_gpu(im3)
            p = self.encoder_p(im_p_)  # keys: NxC
            p = self.head_p(p)
            p = nn.functional.normalize(p, dim=1)  # already normalized
            # # undo shuffle
            p = self._batch_unshuffle_single_gpu(p, idx_unshufflek)

            im_r_, idx_unshuffle = self._batch_shuffle_single_gpu(im4)
            r = self.encoder_k(im_r_)  # keys: NxC
            r = self.head_k(r)
            r = nn.functional.normalize(r, dim=1)  # already normalized
            # undo shuffle
            r = self._batch_unshuffle_single_gpu(r, idx_unshuffle)

        ############MSVQ#################
        logits_qk = torch.einsum('nc,ck->nk', [q, self.queue_k.clone().detach()])
        logits_qp = torch.einsum('nc,ck->nk', [q, self.queue_p.clone().detach()])

        logits_k = torch.einsum('nc,ck->nk', [k, self.queue_k.clone().detach()])
        logits_p = torch.einsum('nc,ck->nk', [p, self.queue_p.clone().detach()])
        logits_r = torch.einsum('nc,ck->nk', [r, self.queue_k.clone().detach()])
        loss_k = - torch.sum(F.softmax(logits_k.detach() / self.tem, dim=1) * F.log_softmax(logits_qk / 0.1, dim=1), dim=1).mean()
        loss_p = - torch.sum(F.softmax(logits_p.detach() / self.tem, dim=1) * F.log_softmax(logits_qp / 0.1, dim=1), dim=1).mean()
        loss_r = - torch.sum(F.softmax(logits_r.detach() / self.tem, dim=1) * F.log_softmax(logits_qk / 0.1, dim=1), dim=1).mean()
        self._dequeue_and_enqueue(k, p)
        return (loss_k+loss_p+loss_r)/3