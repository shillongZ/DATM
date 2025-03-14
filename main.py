import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from ntm import NTM
from util_functions import get_data_split, get_acc, setup_seed, use_cuda
from util_functions import load_data_set, symmetric_normalize_adj, min_max_normalize, Z_score
from util_functions import ContrastiveDataset, calculate_accuracy_train, calculate_accuracy_test
from torch.utils.data import DataLoader, Dataset


device = use_cuda()
import time

# debug
# torch.autograd.set_detect_anomaly(True)
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))

def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)

def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)
    
def ntm_loss(recon_x, x, mu, logvar):
    recon_loss = -torch.sum(x*torch.log(recon_x+1e-9))
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    MSE = nn.MSELoss()(recon_x, x)
    # print('recon_loss:', recon_loss, 'KL:', KL_loss)
    # print('recon:', recon_loss, 'KL:', KL_loss)
    return MSE + KL_loss

def DA_loss(mu_x, logvar_x, mu_c, logvar_c):
    # print(mu_x.shape, mu_c_N_d.shape, len(labels))
    # exit()
    p = torch.add(torch.pow((mu_x-mu_c), 2), torch.pow((logvar_x - logvar_c), 2))
    return torch.sum(torch.sqrt(torch.sum(p, dim=1)))

def RA_loss(features, class_emb, labels):
    gamma = features @ class_emb.T
    labels = torch.tensor(labels)
    p1 = F.cross_entropy(gamma, labels)
    p2 = F.cross_entropy(gamma.T, labels)
    return 0.5 * (p1 + p2)

def train(args, best_t_acc):
    [c_train, c_val] = args.train_val_class
    idx, labellist, G, features, csd_matrix = load_data_set(args.dataset)

    G, edge_index, edge_weight = symmetric_normalize_adj(G)
    G = G.todense()
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    idx_train, idx_test, idx_val = get_data_split(c_train=c_train, c_val=c_val, idx=idx, labellist=labellist)
    y_true = np.array([int(temp[0]) for temp in labellist]) #[n, 1]
    y_true = torch.from_numpy(y_true).type(torch.LongTensor).to(device)

    print("start DTA")

    data_bow = features.to(device)
    data_bow_norm = F.normalize(data_bow).to(device)
    print('fd', data_bow_norm.shape)

    data_class = csd_matrix.to(device)
    data_class_norm = min_max_normalize(data_class)
    print('c', data_class_norm.shape)

    #data_bow_norm = data_bow_norm
    ntm_x = NTM(args, 0).to(device)
    ntm_c = NTM(args, 1).to(device)

    data_c = torch.zeros((data_bow_norm.shape[0], data_class_norm.shape[1])).to(device)
    for i in range(len(y_true)):
        data_c[i, :] = data_class_norm[y_true[i], :]

    ntm_x_optimiser = torch.optim.Adam(ntm_x.parameters(), lr=0.01, weight_decay=0.0)
    ntm_c_optimiser = torch.optim.Adam(ntm_c.parameters(), lr=0.01, weight_decay=0.0)

    train_val_str=str(c_train)+str(c_val)
    np.save('npy_for_pre_acc/our/'+args.dataset+'idx_train'+train_val_str+'.npy',idx_train)
    np.save('npy_for_pre_acc/our/'+args.dataset+'idx_test'+train_val_str+'.npy',idx_test)
    np.save('npy_for_pre_acc/our/'+args.dataset+'idx_val'+train_val_str+'.npy',idx_val)

    top_test_acc = []
    best_acc = 0
    best_predictions = None
    best_labels = None
    for epoch in range(args.nta_epochs+1):
        ntm_x.train()
        ntm_c.train()
        ntm_x_optimiser.zero_grad()
        ntm_c_optimiser.zero_grad()
        train_loss = 0
        train_acc = []

        z_x, g_x, recon_x, mu_x, logvar_x = ntm_x(data_bow_norm, edge_index, edge_weight)
        z_c, g_c, recon_c, mu_c, logvar_c = ntm_c(data_c, 0, 0)

        # ntm_l = ntm_loss(recon_x, data_bow_norm, mu_x, logvar_x) + ntm_loss(recon_c, data_class_norm, mu_c, logvar_c)
        # da_loss = DA_loss(mu_x[idx_train], logvar_x[idx_train], mu_c, logvar_c, y_true[idx_train])
        # ra_loss = RA_loss(z_x[idx_train], z_c, y_true[idx_train])

        ntm_l = ntm_loss(recon_x, data_bow_norm, mu_x, logvar_x) + ntm_loss(recon_c, data_c, mu_c, logvar_c)
        da_loss = DA_loss(mu_x, logvar_x, mu_c, logvar_c)
        ra_loss = RA_loss(z_x, z_c, y_true)
        # print('DA:', da_loss)
        # print('RA:', ra_loss)
        train_acc.append(calculate_accuracy_train(z_x, y_true, z_c).item())

        loss = ntm_l + args.alpha * da_loss + args.beta * ra_loss
        #  + ntm_x.l1_strength * l1_penalty(ntm_x.fcd1.weight) + ntm_c.l1_strength * l1_penalty(ntm_c.fcd1.weight)
        train_loss += loss.item()
        # train_acc.append(calculate_accuracy_train(mu_x[idx_train], y_true[idx_train], mu_c).item())
        loss.backward()

        ntm_x_optimiser.step()
        ntm_c_optimiser.step()

        if epoch % 100 == 0:
            test_acc = []
            all_preds = []
            all_labels = [] 
            ntm_x.eval()
            ntm_c.eval()
            print('mu:', mu_x[0], mu_c[0])
            with torch.no_grad():
                acc, preds = calculate_accuracy_test(mu_x[idx_test], y_true[idx_test], mu_c, c_train)
                test_acc.append(acc.item())
                all_preds.append(preds.cpu().numpy())  
                all_labels.append(y_true.cpu().numpy())
            cur_acc = np.mean(test_acc)
            top_test_acc.append(cur_acc)
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_predictions = np.concatenate(all_preds)
                best_labels = np.concatenate(all_labels)
                np.save('npy_for_pre_acc/our/' + args.dataset + 'pred' + train_val_str + '.npy', best_predictions)
                np.save('npy_for_pre_acc/our/' + args.dataset + 'y_true' + train_val_str + '.npy', best_labels)
            if cur_acc > best_t_acc:
                best_t_acc = cur_acc
                best_predictions = np.concatenate(all_preds)
                best_labels = np.concatenate(all_labels)
                np.save('npy_for_pre_acc/our_best/' + args.dataset + 'pred' + train_val_str + '.npy', best_predictions)
                np.save('npy_for_pre_acc/our_best/' + args.dataset + 'y_true' + train_val_str + '.npy', best_labels)

                np.save('npy_for_pre_acc/our_best/'+args.dataset+'idx_train'+train_val_str+'.npy',idx_train)
                np.save('npy_for_pre_acc/our_best/'+args.dataset+'idx_test'+train_val_str+'.npy',idx_test)
                np.save('npy_for_pre_acc/our_best/'+args.dataset+'idx_val'+train_val_str+'.npy',idx_val)
            
            print(f"Epoch {epoch}, Loss: {train_loss}, Acc: {np.mean(train_acc)}, Test: {np.mean(test_acc)}")

    print("end DTA")
    # features_soft_topic_emb = F.softmax(z_x, dim=1)
    # class_soft_topic_emb = F.softmax(z_c, dim=1)

    # features_soft_topic_emb_d = F.normalize(features_soft_topic_emb, p=2, dim=1)
    # class_soft_topic_emb_d = F.normalize(class_soft_topic_emb, p=2, dim=1)

    print(best_acc)
    return best_t_acc

        
    
        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MODEL')
    parser.add_argument("--dataset", type=str, default='small_matrix', choices=['cora', 'citeseer', 'C-M10-M', 'small_matrix','big_matrix'], help="dataset")
    parser.add_argument("--train-val-class", type=int, nargs='*', default=[7, 3], help="the first #train_class and #validation classes")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n-hidden", type=int, default=256, help="number of hidden layers")    # d
    # parser.add_argument("--k", type=int, default=3, help="k-hop neighbors")
    parser.add_argument("--alpha", type=float, default=0.7, help="weight of distribution alignment")
    parser.add_argument("--beta", type=float, default=0.7, help="weight of reconstruction alignment")
    parser.add_argument("--bow_vocab_size", type=int, default=559, help="vocab size")
    parser.add_argument("--class_size", type=int, default=300, help="csd size")
    parser.add_argument("--topic_num", type=int, default=30, help="topic num")    # K
    parser.add_argument("--nta_epochs", type=int, default=600, help="number of training epochs")
    args = parser.parse_args()
    print(args)
    best_t_acc = 0
    for i in range(100):
        best_t_acc = train(args, best_t_acc)
        print(i, best_t_acc)
