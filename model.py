
"""VSE model"""

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from torch.nn.functional import avg_pool1d, max_pool1d


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.embed_size = opt.embed_size
        self.fc = nn.Linear(opt.img_dim, opt.embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        features = features.permute(0, 2, 1)
        # features = avg_pool1d(features, features.size(2)).squeeze(2)
        features = max_pool1d(features, features.size(2)).squeeze(2)

        features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.embed_size = opt.embed_size

        # word embedding
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)

        # caption embedding
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out, dim=-1)

        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = cosine_sim(im, s)

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class LGLoss(nn.Module):
    """
    Language Guided Loss
    """
    def __init__(self, opt):
        super(LGLoss, self).__init__()

        '''hyper-parameters'''
        self.opt = opt
        self.margin = opt.margin

        percent_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        if 'f30k' in opt.data_name:
            if opt.language_model == 'clip':
                alpha_list = [-1.0, 0.28, 0.33, 0.36, 0.39, 0.42, 0.45, 0.49, 0.52, 0.57, 1.0]
            if opt.language_model == 'mpnet':
                alpha_list = [-1.0, 0.0, 0.04, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 0.28, 1.0]
            if opt.language_model == 'glove':
                alpha_list = [-1.0, 0.43, 0.49, 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.73, 1.0]
        elif 'coco' in opt.data_name:
            if opt.language_model == 'clip':
                alpha_list = [-1.0, 0.31, 0.35, 0.38, 0.41, 0.44, 0.47, 0.49, 0.53, 0.58, 1.0]
            if opt.language_model == 'mpnet':
                alpha_list = [-1.0, 0.03, 0.06, 0.09, 0.11, 0.14, 0.16, 0.19, 0.24, 0.31, 1.0]
            if opt.language_model == 'glove':
                alpha_list = [-1.0, 0.33, 0.38, 0.42, 0.45, 0.48, 0.51, 0.55, 0.58, 0.64, 1.0]
        alpha_id = percent_list.index(opt.percent)
        self.alpha = alpha_list[alpha_id]
        print('alpha:', self.alpha)
        if self.alpha == 1.0:
            self.beta = 1
        else:
            self.beta = self.margin / (1 - self.alpha)
        self.tau = opt.tau

        self.batch_size = opt.batch_size
        self.pos = torch.eye(self.batch_size)
        self.pos = self.pos.cuda()
        self.pos_mask = self.pos > .5
        self.neg = 1 - self.pos

    def forward(self, v, t, v_text_emb, t_text_emb):

        batch_size = v.size(0)

        scores = cosine_sim(v, t)
        pos_scores = scores.diag()

        if batch_size != self.batch_size:
            pos = torch.eye(scores.size(0))
            pos = pos.cuda()
            pos_mask = pos > .5
            neg = 1 - pos
        else:
            pos_mask = self.pos_mask
            neg = self.neg

        '''measure relevance degrees'''
        v_text_emb = Variable(v_text_emb, volatile=False)
        t_text_emb = Variable(t_text_emb, volatile=False)

        if torch.cuda.is_available():
            t_text_emb = t_text_emb.cuda()
            v_text_emb = v_text_emb.cuda()

        v_text_emb = v_text_emb.transpose(0, 1)
        t_text_emb = t_text_emb.view(1, t_text_emb.size(0), t_text_emb.size(1))
        t_text_emb = t_text_emb.expand(5, t_text_emb.size(1), t_text_emb.size(2))

        v_text_emb = l2norm_3d(v_text_emb)
        t_text_emb = l2norm_3d(t_text_emb)
        relevance = torch.bmm(v_text_emb, t_text_emb.transpose(1, 2))
        relevance = relevance.max(0)[0]
        relevance = relevance.masked_fill_(pos_mask, 1)

        margin = 1 - relevance
        margin = margin * self.beta
        margin = torch.where(relevance >= self.alpha, margin, torch.full_like(relevance, self.margin))

        exp_pos_scores = torch.exp(pos_scores / self.tau)
        all_scores = scores + margin
        exp_all_scores = torch.exp(all_scores / self.tau)
        sum_exp_all_scores_0 = torch.sum(exp_all_scores, dim=0)
        sum_exp_all_scores_1 = torch.sum(exp_all_scores, dim=1)
        loss_0 = - torch.log(exp_pos_scores / sum_exp_all_scores_0)
        loss_1 = - torch.log(exp_pos_scores / sum_exp_all_scores_1)
        loss_0 = loss_0.mean()
        loss_1 = loss_1.mean()
        loss = loss_0 + loss_1

        return loss


class VSE(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt)
        self.txt_enc = EncoderText(opt)

        print(self.img_enc)
        print(self.txt_enc)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.loss = opt.loss
        if self.loss == 'contrastive':
            self.criterion = ContrastiveLoss(opt=opt,
                                             margin=opt.margin,
                                             max_violation=opt.max_violation)
        elif self.loss == 'lg':
            self.criterion = LGLoss(opt)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        # images = Variable(images)
        # captions = Variable(captions)
        images = images.cuda()
        captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, v_bert_emb, t_bert_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        if self.loss in ['contrastive']:
            loss = self.criterion(img_emb, cap_emb)
            self.logger.update('Loss', loss.item(), img_emb.size(0))
        else:
            loss = self.criterion(img_emb, cap_emb, v_bert_emb, t_bert_emb)
            self.logger.update('Loss', loss.item(), img_emb.size(0))

        return loss

    def train_emb(self, images, captions, lengths, image_ids, caption_ids, v_text_emb, t_text_emb, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, v_text_emb, t_text_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
