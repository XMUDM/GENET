
import sys

from torch.optim.lr_scheduler import StepLR

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

from data.dataset import Gowalla, Foursquare, Amazon, GowallaUCS, FoursquareUCS, AmazonUCS, AmazonICS
from models.HGNNP import HGNNP
from dhg.random import set_seed
from dhg.nn import BPRLoss, EmbeddingRegularization
import logging
import os
from torch.utils.data import DataLoader

from models.Signal import GCFSignal
from utils.batch_test import test_united
import time
import torch
import torch.nn.functional as F
from utils.visualize import embedding_visualize
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
    # ui_bigraph_hat = ui_bigraph
    for anchor, pos, neg in data_loader:

        emb =  net_ds(u_embs,i_embs,ui_bigraph_hat)

        pos_scores, neg_scores = (emb[anchor] * emb[pos]).sum(dim=1),\
                                           (emb[anchor] * emb[neg]).sum(dim=1)

        emb_norm = F.normalize(emb, dim=-1)
        # print(emb_norm.size())
        pos_u_intra = torch.exp(
            torch.sum(emb_norm[anchor] * emb_norm[anchor], dim=-1) / 0.2)  # [batch_size]
        neg_u_intra = torch.sum(torch.exp(
            torch.matmul(emb_norm[anchor], emb_norm[anchor].transpose(1, 0)) / 0.2),
            dim=-1)  # [batch_size]
        loss_con = -torch.mean(torch.log(pos_u_intra / (pos_u_intra + neg_u_intra)))
        pos_u_intra = torch.exp(
            torch.sum(emb_norm[pos] * emb_norm[pos], dim=-1) / 0.2)  # [batch_size]x
        neg_u_intra = torch.sum(torch.exp(
            torch.matmul(emb_norm[pos], emb_norm[pos].transpose(1, 0)) / 0.2),
            dim=-1)  # [batch_size]
        loss_con2 = -torch.mean(torch.log(pos_u_intra / (pos_u_intra + neg_u_intra)))
        loss1 = criterion(pos_scores,neg_scores)

        loss = loss1+cl_coef*loss_con+cl_coef*loss_con2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
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
    work_root = 'your work root'
    assert work_root != 'your work root', 'please set your work root'

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='foursquare')
    parser.add_argument("--pretrain_model", type=str, default="500")
    args = parser.parse_args()

    with open('../config/prompt.json','r') as f:
        config = json.load(f)
    set_seed(config['seed'])
    seed_torch(config['seed'])
    dim_emb = config['dim_emb']
    batch_sz = config['batch_sz']
    cl_coef = config['cl_coef']
    epochs = config['epochs']
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    val_freq = config['val_freq']
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s]','%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    curtime = datetime.date.today()
    if not os.path.exists(os.path.join(work_root+'exp_cache/',args.dataset+"-"+args.pretrain_model)):
        os.makedirs(os.path.join(work_root+'exp_cache/',args.dataset+"-"+args.pretrain_model))
    logfile = logging.FileHandler(os.path.join(work_root+'exp_cache/',args.dataset+"-"+args.pretrain_model,str(curtime)+"finetuning_log.txt"))
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    if args.dataset == 'gowalla':
        data = Gowalla(data_root=work_root+'data/')
    elif args.dataset == 'foursquare':
        data = Foursquare(data_root=work_root+'data/')
    elif args.dataset == 'amazon':
        data = Amazon(data_root=work_root+'data/')
    else:
        raise ValueError("dataset must be gowalla, foursquare or amazon")
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

    #
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
    # train_edge_list_forhg = adj_list_to_edge_list(train_list_forhg)

    test_edge_list = adj_list_to_edge_list(test_list)

    use_hypergraph_update = True
    if use_hypergraph_update:
        if os.path.exists('../save/model/'+args.dataset+'/node_hedge_inci_matrix_plus.pth'):
            node_hedge_inci_matrix = torch.load('../save/model/'+args.dataset+'/node_hedge_inci_matrix_plus.pth').to(device)
        else:
            indices = []
            hy_dict = {}
            for i in range(num_u + num_i):
                hy_dict[i] = hg.N_e(i)
            for edge in train_edge_list:
                for hyedge in hy_dict[edge[0]]:
                    indices.append([edge[1], hyedge.cpu()])
                for hyedge in hy_dict[edge[1]]:
                    indices.append([edge[0], hyedge.cpu()])
            indices = torch.tensor(indices)
            rows = indices[:, 0]
            cols = indices[:, 1]
            values = torch.ones(len(indices))
            inter_node_hedge_inci_matrix = torch.sparse.FloatTensor(torch.stack([rows, cols]), values,
                                                                    torch.Size([num_u + num_i, len(hg.e[0])])).to(device)

            node_hedge_inci_matrix = hg.H + inter_node_hedge_inci_matrix
            torch.save(node_hedge_inci_matrix, '../save/model/'+args.dataset+'/node_hedge_inci_matrix_plus.pth')
    else:
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

    params = list(net_ds.parameters())+[u_embs,i_embs]
    data_loader = 0

    optimizer = optim.Adam(params,lr=config['lr']*10)

    criterion = BPRLoss()
    degrees = torch.zeros(dim_features+dim_features_item)
    for src, dst in train_list:
        degrees[src] += 1
        degrees[dst] += 1
    probs = []
    for src, dst in train_list:
        probs.append(1 / (torch.sqrt(degrees[src]) * torch.sqrt(degrees[dst])))
    probs = torch.FloatTensor(probs)


    train_inter_num = len(train_edge_list)
    train_dataset = UserItemDataset(dim_features,dim_features_item,train_edge_list)
    # test_dataset = UserItemDataset(dim_features,dim_features_item,train_list,train_user_item_list=test_list,phase='test')
    test_dataset = UserItemDataset(dim_features,dim_features_item,test_edge_list,train_user_item_list=train_list,phase='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)


    testRatings, testNegatives = data.test_items,data.neg_items
    print("train start")
    for epoch in range(1,epochs+1):
        if epoch > 3 :
            optimizer.param_groups[0]['lr'] = config['lr']
        train(net_ds, train_loader, optimizer, epoch, criterion)

        net_ds.eval()
        with torch.no_grad():
            embs = net_ds(u_embs, i_embs,ui_bigraph)
        ret = test_united(embs, testRatings, testNegatives)
        perf_str = 'Dataset:%s, Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (args.dataset,
            epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
            ret['ndcg'][2])
        logger.info(perf_str)





