import sys

from matplotlib import pyplot as plt

sys.path.append("..")
import argparse
import gc
import json
from copy import deepcopy

import dhg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime

from dhg import Hypergraph, BiGraph
from dhg.utils import adj_list_to_edge_list, edge_list_to_adj_dict
from data.dataset_wrapers import UserItemDataset

from data.dataset_wrapers import HyperDataset
from data.dataset import Gowalla, Foursquare, Amazon, GowallaICS, GowallaUCS, FoursquareUCS, AmazonUCS
from models.HGNNP import HGNNP, HGNNP_NoLight
from dhg.random import set_seed
from dhg.experiments import HypergraphVertexClassificationTask as Task
from dhg.metrics import UserItemRecommenderEvaluator as Evaluator
from dhg.nn import BPRLoss, EmbeddingRegularization
import logging
import os
from torch.utils.data import DataLoader

from models.Signal import GCFSignal
from utils.batch_test import test_united
import time
import torch.nn.functional as F
import random


def keep_edges(train_list, probs,pruning_ratio):

    num_removed = int((1-pruning_ratio) * len(train_list))
    perserve_idx = torch.multinomial(probs, num_removed, replacement=False)

    perserve_list = [[train_list[i][0],train_list[i][1]] for i in perserve_idx]
    return perserve_list

def train(net_ds,data_loader, optimizer, epoch, criterion):
    net_ds.train()
    loss_mean = 0

    ui_bigraph_hat = ui_bigraph.from_adj_list(dim_features,dim_features_item,
                                              keep_edges(gcn_train_list, probs, pruning_ratio=0.5)).to(device)

    for anchor, pos, neg in data_loader:

        emb =  net_ds(u_embs,i_embs,ui_bigraph_hat)

        pos_scores, neg_scores = (emb[anchor] * emb[pos]).sum(dim=1),\
                                           (emb[anchor] * emb[neg]).sum(dim=1)

        emb_norm = F.normalize(emb, dim=-1)
        pos_u_intra = torch.exp(
            torch.sum(emb_norm[anchor] * emb_norm[anchor], dim=-1) / 0.2)  # [batch_size]
        neg_u_intra = torch.sum(torch.exp(
            torch.matmul(emb_norm[anchor], emb_norm[anchor].transpose(1, 0)) / 0.2),
            dim=-1)  # [batch_size]
        loss_con = -torch.mean(torch.log(pos_u_intra / (pos_u_intra + neg_u_intra)))
        pos_u_intra = torch.exp(
            torch.sum(emb_norm[pos] * emb_norm[pos], dim=-1) / 0.2)  # [batch_size]
        neg_u_intra = torch.sum(torch.exp(
            torch.matmul(emb_norm[pos], emb_norm[pos].transpose(1, 0)) / 0.2),
            dim=-1)  # [batch_size]
        loss_con2 = -torch.mean(torch.log(pos_u_intra / (pos_u_intra + neg_u_intra)))
        loss1 = criterion(pos_scores,neg_scores)

        loss = loss1+cl_coef*loss_con+cl_coef*loss_con2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mean += loss.item()
    loss_mean /= len(data_loader)
    logger.info(f"Epoch:{epoch}, Loss:{loss_mean:.5f}")
    gc.collect()

def seed_torch(seed=2022):
    print("Random seed：",seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    work_root = "your root path"
    assert work_root != "your root path", "please set your work root path"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='foursquare')
    parser.add_argument("--pretrain_model", type=str, default="")
    args = parser.parse_args()

    with open('../config/finetune.json','r') as f:
        config = json.load(f)
    set_seed(config['seed'])
    seed_torch(config['seed'])
    dim_emb = config['dim_emb']
    batch_sz = config['batch_sz']
    cl_coef = config['cl_coef']
    epochs = config['epochs']
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    val_freq = config['val_freq']
    #  set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s]','%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    curtime = datetime.date.today()
    logfile = logging.FileHandler(os.path.join(work_root+'exp_cache/'+f"{args.dataset}-{args.pretrain_model}/"+str(curtime)+"_usercoldstart.txt"))
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    if args.dataset == 'gowalla':
        data = GowallaUCS(data_root=work_root+'data/')
    elif args.dataset == 'foursquare':
        data = FoursquareUCS(data_root=work_root+'data/')
    elif args.dataset == 'amazon':
        data = AmazonUCS(data_root=work_root+'data/')
    else:
        raise ValueError("dataset must be gowalla,foursquare and amazon")
    dim_features = data["num_user_vertices"]
    dim_features_item = data["num_item_vertices"]
    num_classes = data["num_classes"]
    train_list = data['train_list']
    test_list = data['test_list']
    hg = dhg.Hypergraph.load('../save/structure/'+args.dataset+'/hg_plus.pkl').to(device)
    print("hypergraph shape:",hg.H.shape)

    X = torch.sparse_coo_tensor(torch.arange(dim_features + dim_features_item).unsqueeze(0).repeat(2, 1),
                                torch.ones(dim_features + dim_features_item),
                                torch.Size([dim_features + dim_features_item, dim_features + dim_features_item])).to(
        device)

    net = HGNNP(X.shape[1],dim_emb,use_bn=True).to(device)

    num_u, num_i = data["num_user_vertices"], data["num_item_vertices"]


    # load pretrained model
    pretrained_dict = torch.load('../save/model/'+args.dataset+f'/HGNNP_plus_LP_PM_{args.pretrain_model}.pth')
    model_dict = net.state_dict()
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


    gcn_train_list = [[user,item-num_u] for user,item in train_list]
    ui_bigraph = BiGraph.from_adj_list(num_u, num_i, gcn_train_list)
    ui_bigraph = ui_bigraph.to(device)

    train_edge_list = adj_list_to_edge_list(train_list)
    test_edge_list = adj_list_to_edge_list(test_list)

    # It is not suitable to update the matrix here, because we use Kshot,
    # or we should only be able to update the matrix of Kshot,
    # but it is not necessary to update, because finetuning with LightGCN is enough.
    node_hedge_inci_matrix = hg.H



    net.eval()
    with torch.no_grad():
        nodes_emb, hedges_emb = net(X, hg)
    nodes_emb = torch.nn.functional.normalize(nodes_emb, dim=-1).detach()
    hedges4node_emb = torch.nn.functional.normalize(torch.matmul(node_hedge_inci_matrix,
                                                                 hedges_emb), dim=-1).detach()
    merge_embs = nodes_emb+hedges4node_emb

    net_ds = GCFSignal(num_layers=2).to(device)

    u_embs = torch.nn.Parameter(merge_embs[:num_u],requires_grad=True)
    i_embs = torch.nn.Parameter(merge_embs[num_u:],requires_grad=True)

    # Calculate the probability of each node being deleted.
    degrees = torch.zeros(dim_features+dim_features_item)
    for src, dst in train_list:
        degrees[src] += 1
        degrees[dst] += 1
    probs = []
    for src, dst in train_list:
        probs.append(1 / (torch.sqrt(degrees[src]) * torch.sqrt(degrees[dst])))
    probs = torch.FloatTensor(probs)

    params = list(net_ds.parameters())+[u_embs,i_embs]
    data_loader = 0
    optimizer = optim.Adam(params,lr=config['lr']*10)
    criterion = BPRLoss()


    train_inter_num = len(train_edge_list)
    train_dataset = UserItemDataset(dim_features,dim_features_item,train_edge_list)
    # test_dataset = UserItemDataset(dim_features,dim_features_item,train_list,train_user_item_list=test_list,phase='test')
    test_dataset = UserItemDataset(dim_features,dim_features_item,test_edge_list,train_user_item_list=train_list,phase='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)


    testRatings, testNegatives = data.test_items,data.neg_items
    best_state, best_val, best_epoch = None, 0, -1
    print("train start")
    recall_10_list = []
    for epoch in range(1,epochs+1):
        if epoch > 3:
            optimizer.param_groups[0]['lr'] = config['lr']
        train(net_ds, train_loader, optimizer, epoch, criterion)
        net_ds.eval()
        with torch.no_grad():
            embs = net_ds(u_embs,i_embs,ui_bigraph)
            # embedding_visualize(embs, num_u, num_i, f"embedding:{epoch}")
        ret = test_united(embs, testRatings, testNegatives)
        perf_str = 'Dataset:%s, Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (args.dataset,

            epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
            ret['ndcg'][2])
        recall_10_list.append(ret['recall'][0])
        logger.info(perf_str)
        if ret['recall'][0] > best_val:
            best_epoch = epoch
            best_val = ret['recall'][0]
            best_state = deepcopy(net_ds.state_dict())

    plt.title(f"{args.dataset} user coldstart")
    plt.plot(recall_10_list,marker='o')
    plt.xlabel("epoch")
    plt.ylabel("recall@10")
    plt.show()





