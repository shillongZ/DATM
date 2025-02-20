import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from util_functions import get_data_split, get_acc, setup_seed, use_cuda
from torch_geometric.nn import GCNConv

device = use_cuda()

class NTM(nn.Module):
    def __init__(self, opt, flag, l1_strength=0.001):
        super(NTM, self).__init__()
        hidden_dim = opt.n_hidden
        self.topic_num = opt.topic_num
        topic_num = opt.topic_num
        self.flag = flag
        if flag == 0:
            self.input_dim = opt.bow_vocab_size
            self.gc11 = GCNConv(self.input_dim, hidden_dim)
            self.gc12 = GCNConv(hidden_dim, hidden_dim)
        elif flag == 1:
            self.input_dim = opt.class_size
            self.fc11 = nn.Linear(self.input_dim, hidden_dim)
            self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim)
        self.l1_strength = torch.FloatTensor([l1_strength]).to(device)

    def encode(self, x, edge_index, edge_weight):
        if self.flag == 0:
            e1 = F.relu(self.gc11(x, edge_index, edge_weight))
            e1 = F.relu(self.gc12(e1, edge_index, edge_weight))
        elif self.flag == 1:
            e1 = F.relu(self.fc11(x))
            e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x))
        return self.fc21(e1), self.fc22(e1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1

    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=1)
        return d1

    def forward(self, x, edge_index, edge_weight):
        mu, logvar = self.encode(x.view(-1, self.input_dim), edge_index, edge_weight)  # logvar = log sigma^2
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        return z, g, self.decode(g), mu, logvar

    def print_topic_words(self, vocab_dic, fn, n_top_words=10):
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        logging.info("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()
# h=256
# class NTM(nn.Module):
#     def __init__(self, opt, hidden_dim=256, l1_strength=0.001):
#         super(NTM, self).__init__()
#         self.input_bow = opt.bow_vocab_size
#         self.input_class = opt.class_size
#         self.topic_num = opt.topic_num
#         topic_num = opt.topic_num
#         self.fc11_bow = nn.Linear(self.input_bow, hidden_dim)
#         self.fc11_class = nn.Linear(self.input_class, hidden_dim)
#         self.fc12 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, topic_num)
#         self.fc22 = nn.Linear(hidden_dim, topic_num)
#         self.fcs_bow = nn.Linear(self.input_bow, hidden_dim, bias=False)
#         self.fcs_class = nn.Linear(self.input_class, hidden_dim, bias=False)
#         self.fcg1 = nn.Linear(topic_num, topic_num)
#         self.fcg2 = nn.Linear(topic_num, topic_num)
#         self.fcg3 = nn.Linear(topic_num, topic_num)
#         self.fcg4 = nn.Linear(topic_num, topic_num)
#         self.fcd1_bow = nn.Linear(topic_num, self.input_bow)
#         self.fcd1_class = nn.Linear(topic_num, self.input_class)
#         self.l1_strength = torch.FloatTensor([l1_strength]).to(device)

#     def encode(self, x, data_type):
#         if data_type == 'bow':
#             e1 = F.relu(self.fc11_bow(x))
#         elif data_type == 'class':
#             e1 = F.relu(self.fc11_class(x))
#         e1 = F.relu(self.fc12(e1))
#         if data_type == 'bow':
#             e1 = e1.add(self.fcs_bow(x))
#         elif data_type == 'class':
#             e1 = e1.add(self.fcs_class(x))
#         return self.fc21(e1), self.fc22(e1)

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def generate(self, h):
#         g1 = torch.tanh(self.fcg1(h))
#         g1 = torch.tanh(self.fcg2(g1))
#         g1 = torch.tanh(self.fcg3(g1))
#         g1 = torch.tanh(self.fcg4(g1))
#         g1 = g1.add(h)
#         return g1

#     def decode(self, z, data_type):
#         if data_type == 'bow':
#             d1 = F.softmax(self.fcd1_bow(z), dim=1)
#         elif data_type == 'class':
#             d1 = F.softmax(self.fcd1_class(z), dim=1)
#         return d1

#     def forward(self, x, data_type):
#         if data_type == 'bow':
#             input_dim = self.input_bow
#         elif data_type == 'class':
#             input_dim = self.input_class
#         mu, logvar = self.encode(x.view(-1, input_dim), data_type)
#         z = self.reparameterize(mu, logvar)
#         g = self.generate(z)
#         return z, g, self.decode(g, data_type), mu, logvar

#     def print_topic_words(self, vocab_dic, fn, n_top_words=10):
#         beta_exp = self.fcd1.weight.data.cpu().numpy().T
#         logging.info("Writing to %s" % fn)
#         fw = open(fn, 'w')
#         for k, beta_k in enumerate(beta_exp):
#             topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
#             print('Topic {}: {}'.format(k, ' '.join(topic_words)))
#             fw.write('{}\n'.format(' '.join(topic_words)))
#         fw.close()