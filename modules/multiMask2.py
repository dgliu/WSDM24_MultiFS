import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, statistics
from modules.layers import FactorizationMachine, MultiLayerPerceptron
import copy
import modules.layers as layer


class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)


class MaskEmbedding(nn.Module):
    def __init__(self, feature_num, latent_dim, mask_weight_init_value=0., mask_scaling=1):
        super().__init__()
        self.feature_num = feature_num
        self.latent_dim = latent_dim
        self.mask_weight_init_value = mask_weight_init_value
        self.mask_scaling = mask_scaling
        self.embedding = nn.Parameter(torch.zeros(feature_num, latent_dim))
        nn.init.xavier_uniform_(self.embedding)
        self.init_weight = nn.Parameter(torch.zeros_like(self.embedding), requires_grad=False)
        self.init_mask()
        self.sign = LBSign.apply

    def init_mask(self):
        self.mask_weight_i = nn.Parameter(torch.Tensor(self.feature_num, 1))
        self.mask_weight_s = nn.Parameter(torch.Tensor(self.feature_num, 1))
        self.mask_weight_j = nn.Parameter(torch.Tensor(self.feature_num, 1))
        nn.init.constant_(self.mask_weight_i, self.mask_weight_init_value)
        nn.init.constant_(self.mask_weight_s, self.mask_weight_init_value)
        nn.init.constant_(self.mask_weight_j, self.mask_weight_init_value)

    def compute_mask(self, x, temp, ticket):
        scaling = self.mask_scaling * (1. / sigmoid(self.mask_weight_init_value))
        mask_weight_i = F.embedding(x, self.mask_weight_i)
        mask_weight_s = F.embedding(x, self.mask_weight_s)
        mask_weight_j = F.embedding(x, self.mask_weight_j)
        if ticket:
            mask_i = (mask_weight_i > 0).float()
            mask_s = (mask_weight_s > 0).float()
            mask_j = (mask_weight_j > 0).float()
        else:
            mask_i = torch.sigmoid(temp * mask_weight_i)
            mask_s = torch.sigmoid(temp * mask_weight_s)
            mask_j = torch.sigmoid(temp * mask_weight_j)

        return scaling * mask_i, scaling * mask_s, scaling * mask_j

    def prune(self, temp):
        self.mask_weight_i.data = torch.clamp(temp * self.mask_weight_i.data, max=self.mask_weight_init_value)
        self.mask_weight_s.data = torch.clamp(temp * self.mask_weight_s.data, max=self.mask_weight_init_value)
        self.mask_weight_j.data = torch.clamp(temp * self.mask_weight_j.data, max=self.mask_weight_init_value)


    def forward(self, x, temp=1, thre=1, ticket=False):
        embed = F.embedding(x, self.embedding)
        mask_i, mask_s, mask_j = self.compute_mask(x, temp, ticket)
        g_s = self.sign(torch.relu(mask_s - thre))
        mask1 = mask_s * g_s + mask_i * (1 - g_s)
        mask2 = mask_s * g_s + mask_j * (1 - g_s)
        return embed * mask_s, embed * mask1, embed * mask2

    def compute_remaining_weights(self, temp, ticket=False):
        if ticket:
            m_s = (self.mask_weight_s > 0.).float()
            m_i = (self.mask_weight_i > 0.).float()
            m_j = (self.mask_weight_j > 0.).float()
            m_si = ((self.mask_weight_s > 0.) & (self.mask_weight_i > 0.)).float()
            m_sj = ((self.mask_weight_s > 0.) & (self.mask_weight_j > 0.)).float()
            return m_s.sum() / m_s.numel(), (m_i - m_si).sum() / self.mask_weight_i.numel(), (m_j - m_sj).sum() / self.mask_weight_j.numel()
        else:
            m_s = torch.sigmoid(temp * self.mask_weight_s)
            m_i = torch.sigmoid(temp * self.mask_weight_i)
            m_j = torch.sigmoid(temp * self.mask_weight_j)
            print("max mask weight s: {wa:6f}, min mask weight s: {wi:6f}".format(wa=torch.max(self.mask_weight_s),
                                                                                  wi=torch.min(self.mask_weight_s)))
            print("max mask s: {ma:8f}, min mask s: {mi:8f}".format(ma=torch.max(m_s), mi=torch.min(m_s)))
            print("mask s number: {mn:6f}".format(mn=float((m_s == 0.).sum())))
            print("max mask weight i: {wa:6f}, min mask weight i: {wi:6f}".format(wa=torch.max(self.mask_weight_i),
                                                                                  wi=torch.min(self.mask_weight_i)))
            print("max mask i: {ma:8f}, min mask i: {mi:8f}".format(ma=torch.max(m_i), mi=torch.min(m_i)))
            print("mask i number: {mn:6f}".format(mn=float((m_i == 0.).sum())))
            print("max mask weight j: {wa:6f}, min mask weight j: {wi:6f}".format(wa=torch.max(self.mask_weight_j),
                                                                                  wi=torch.min(self.mask_weight_j)))
            print("max mask j: {ma:8f}, min mask j: {mi:8f}".format(ma=torch.max(m_j), mi=torch.min(m_j)))
            print("mask j number: {mn:6f}".format(mn=float((m_j == 0.).sum())))
            return None

    def checkpoint(self):
        self.init_weight.data = self.embedding.clone()

    def rewind_weights(self):
        self.embedding.data = self.init_weight.clone()

    def reg1_s(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight_s))

    def reg1_i(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight_i))

    def reg1_j(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight_j))

    def reg2_ij(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight_i) * torch.sigmoid(temp * self.mask_weight_j))


class MaskedNet(nn.Module):
    def __init__(self, opt):
        super(MaskedNet, self).__init__()
        self.ticket = False
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        self.mask_embedding = MaskEmbedding(self.feature_num, self.latent_dim,
                                            mask_weight_init_value=opt["mask_weight_initial"],
                                            mask_scaling=opt["mask_scaling"])
        self.mask_modules = [m for m in self.modules() if type(m) == MaskEmbedding]
        self.temp = 1
        self.thre = nn.Parameter(torch.Tensor([opt["init_thre"] * opt["mask_scaling"]]))

    def checkpoint(self):
        for m in self.mask_modules: m.checkpoint()
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules: m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)

    def prune(self):
        for m in self.mask_modules: m.prune(self.temp)

    def reg1_s(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg1_s(self.temp)
        return reg_loss

    def reg1_i(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg1_i(self.temp)
        return reg_loss

    def reg1_j(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg1_j(self.temp)
        return reg_loss

    def reg2_ij(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg2_ij(self.temp)
        return reg_loss


class MaskDNN(MaskedNet):
    def __init__(self, opt):
        super(MaskDNN, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn0 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embeddings, x_embedding0, x_embedding1 = self.mask_embedding(x, self.temp, self.thre, self.ticket)
        x_dnns = x_embeddings.view(-1, self.dnn_dim)
        x_dnn0 = x_embedding0.view(-1, self.dnn_dim)
        x_dnn1 = x_embedding1.view(-1, self.dnn_dim)
        output_dnn0 = self.dnn0(x_dnn0)
        output_dnn1 = self.dnn1(x_dnn1)
        output_dnn0s = self.dnn0(x_dnns)
        output_dnn1s = self.dnn1(x_dnns)
        logit0 = output_dnn0
        logit1 = output_dnn1
        logit0s = output_dnn0s
        logit1s = output_dnn1s
        return logit0, logit1, logit0s, logit1s

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


class MaskDeepFM(MaskedNet):
    def __init__(self, opt):
        super(MaskDeepFM, self).__init__(opt)
        self.fm0 = FactorizationMachine(reduce_sum=True)
        self.fm1 = FactorizationMachine(reduce_sum=True)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn0 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embeddings, x_embedding0, x_embedding1 = self.mask_embedding(x, self.temp, self.thre, self.ticket)

        output_fm0 = self.fm0(x_embedding0)
        output_fm0s = self.fm0(x_embeddings)
        x_dnn0 = x_embedding0.view(-1, self.dnn_dim)
        x_dnns = x_embeddings.view(-1, self.dnn_dim)
        output_dnn0 = self.dnn0(x_dnn0)
        output_dnn0s = self.dnn0(x_dnns)

        output_fm1 = self.fm1(x_embedding1)
        output_fm1s = self.fm1(x_embeddings)
        x_dnn1 = x_embedding1.view(-1, self.dnn_dim)
        output_dnn1 = self.dnn1(x_dnn1)
        output_dnn1s = self.dnn1(x_dnns)

        logit0 = output_dnn0 + output_fm0
        logit1 = output_dnn1 + output_fm1
        logit0s = output_dnn0s + output_fm0s
        logit1s = output_dnn1s + output_fm1s
        return logit0, logit1, logit0s, logit1s

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


class MaskDeepCross(MaskedNet):
    def __init__(self, opt):
        super(MaskDeepCross, self).__init__(opt)
        self.dnn_dim = self.field_num * self.latent_dim
        cross_num = opt["cross"]
        mlp_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.cross0 = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.cross1 = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn0 = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination0 = nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)
        self.combination1 = nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x):
        x_embeddings, x_embedding0, x_embedding1 = self.mask_embedding(x, self.temp, self.thre, self.ticket)

        x_dnns = x_embeddings.view(-1, self.dnn_dim)
        x_dnn0 = x_embedding0.view(-1, self.dnn_dim)
        x_dnn1 = x_embedding1.view(-1, self.dnn_dim)

        output_cross0 = self.cross0(x_dnn0)
        output_dnn0 = self.dnn0(x_dnn0)
        comb_tensor0 = torch.cat((output_cross0, output_dnn0), dim=1)
        output_cross0s = self.cross0(x_dnns)
        output_dnn0s = self.dnn0(x_dnns)
        comb_tensor0s = torch.cat((output_cross0s, output_dnn0s), dim=1)

        output_cross1 = self.cross1(x_dnn1)
        output_dnn1 = self.dnn1(x_dnn1)
        comb_tensor1 = torch.cat((output_cross1, output_dnn1), dim=1)
        output_cross1s = self.cross1(x_dnns)
        output_dnn1s = self.dnn1(x_dnns)
        comb_tensor1s = torch.cat((output_cross1s, output_dnn1s), dim=1)

        logit0 = self.combination0(comb_tensor0)
        logit1 = self.combination1(comb_tensor1)
        logit0s = self.combination0(comb_tensor0s)
        logit1s = self.combination1(comb_tensor1s)
        return logit0, logit1, logit0s, logit1s


    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)



def getOptim(network, optim, lr, l2):
    weight_params = map(lambda a: a[1],
                        filter(lambda p: p[1].requires_grad and 'mask_weight' not in p[0] and 'thre' not in p[0],
                               network.named_parameters()))
    mask_params = map(lambda a: a[1],
                      filter(lambda p: p[1].requires_grad and 'mask_weight' in p[0], network.named_parameters()))
    thre_params = map(lambda a: a[1],
                      filter(lambda p: p[1].requires_grad and 'thre' in p[0], network.named_parameters()))
    optim = optim.lower()
    if optim == "sgd":
        return [torch.optim.SGD(weight_params, lr=lr, weight_decay=l2), torch.optim.SGD(mask_params, lr=lr),
                torch.optim.SGD(thre_params, lr=0.01 * lr)]
    elif optim == "adam":
        return [torch.optim.Adam(weight_params, lr=lr, weight_decay=l2), torch.optim.Adam(mask_params, lr=lr),
                torch.optim.Adam(thre_params, lr=0.01 * lr)]
    else:
        raise ValueError("Invalid optimizer type: {}".format(optim))


def getModel(model: str, opt):
    model = model.lower()
    if model == "deepfm":
        return MaskDeepFM(opt)
    elif model == "dcn":
        return MaskDeepCross(opt)
    elif model == "dnn":
        return MaskDNN(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))


def sigmoid(x):
    return float(1. / (1. + np.exp(-x)))
