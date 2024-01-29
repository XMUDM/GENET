import argparse
import sys

from torch import optim

sys.path.append("..")
import gc
import json
from copy import deepcopy


import dhg
import torch
import datetime

from dhg import Hypergraph, BiGraph, DiGraph
from dhg.utils import adj_list_to_edge_list, edge_list_to_adj_dict
from data.dataset_wrapers import UserItemDataset, HyperSeqDataset

from data.dataset import GowallaSeq, FoursquareSeq, AmazonSeq
from models.HGNNP import HGNNP
from dhg.random import set_seed
from dhg.metrics import UserItemRecommenderEvaluator as Evaluator
from dhg.nn import BPRLoss, EmbeddingRegularization
import logging
import os
from torch.utils.data import DataLoader
from utils.batch_test import test_united
import time
import torch.nn.functional as F
from models.Signal import  LG2SeqSignal

def train(net, X, A, data_loader, optimizer, epoch, criterion):
    net_prompt.train()
    loss_mean = 0
    for user,user_seq, pos, neg in data_loader:
        user_seq = torch.stack(user_seq).transpose(1,0).to(device)
        h,emb =  net_prompt(u_embs,i_embs,ui_bigraph,user_seq)
        h = emb[user]+h
        pos_scores, neg_scores = (h * emb[pos]).sum(dim=1),\
                                           (h * emb[neg]).sum(dim=1)

        loss3 = criterion(pos_scores,neg_scores)

        loss = loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mean += loss.item()
    loss_mean /= len(data_loader)
    logger.info(f"Epoch:{epoch}, Loss:{loss_mean:.5f}")
    gc.collect()

if __name__ == "__main__":
    work_root = "your root path"
    assert work_root != "your root path" , "please set your work root path"
    # Parse the hyperparameter.
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument("--pretrain_model", type=str, default="500")
    args = parser.parse_args()

    # Read the hyperparameter file.
    with open('../config/finetune_seq.json','r') as f:
        config = json.load(f)

    set_seed(config['seed'])
    dim_emb = config['dim_emb']
    batch_sz = config['batch_sz']
    epochs = config['epochs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_freq = config['val_freq']
    #  set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s]','%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    curtime = datetime.date.today()
    logfile = logging.FileHandler(os.path.join(work_root+'exp_cache/',args.dataset+"-"+args.pretrain_model,str(curtime)+"_finetuning_seq_log.txt"))
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    if args.dataset == 'gowalla':
        data = GowallaSeq(data_root=work_root+'data')
    elif args.dataset == 'foursquare':
        data = FoursquareSeq(data_root=work_root+'data')
    elif args.dataset == 'amazon':
        data = AmazonSeq(data_root=work_root+'data')
    else:
        raise ValueError("dataset must be gowalla, foursquare or amazon")

    dim_features = data["num_user_vertices"]
    dim_features_item = data["num_item_vertices"]
    num_classes = data["num_classes"]
    train_list = data['train_list']
    test_list = data['test_list']
    hg = dhg.Hypergraph.load(f'../save/structure/{args.dataset}/hg_plus.pkl').to(device)
    print("hypergraph shape:",hg.H.shape)

    X = torch.sparse_coo_tensor(torch.arange(dim_features + dim_features_item).unsqueeze(0).repeat(2, 1),
                                torch.ones(dim_features + dim_features_item),
                                torch.Size([dim_features + dim_features_item, dim_features + dim_features_item])).to(
        device)

    net = HGNNP(X.shape[1],dim_emb,use_bn=True).to(device)

    num_u, num_i = data["num_user_vertices"], data["num_item_vertices"]

    # Load a pretrained model.
    pretrained_dict = torch.load(f'../save/model/{args.dataset}/HGNNP_plus_LP_PM_{args.pretrain_model}.pth')
    model_dict = net.state_dict()
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


    train_edge_list = adj_list_to_edge_list(train_list)
    gcn_train_list = [[user,item-num_u] for user,item in train_edge_list]
    ui_bigraph = BiGraph.from_adj_list(num_u, num_i, gcn_train_list)
    ui_bigraph = ui_bigraph.to(device)
    node_hedge_inci_matrix = hg.H

    nodes_emb, hedges_emb = net(X, hg)
    nodes_emb = torch.nn.functional.normalize(nodes_emb, dim=-1).detach()

    hedges4node_emb = torch.nn.functional.normalize(torch.matmul(node_hedge_inci_matrix,
                                                                 hedges_emb), dim=-1).detach()

    net_prompt = LG2SeqSignal(dim_emb,num_layers=2).to(device)

    merge_embs = nodes_emb+hedges4node_emb


    u_embs = torch.nn.Parameter(merge_embs[:num_u],requires_grad=True)
    i_embs = torch.nn.Parameter(merge_embs[num_u:],requires_grad=True)

    # Set the parameters to train.
    params = list(net_prompt.parameters())+[u_embs,i_embs]

    data_loader = 0
    # The initial learning rate is 0.005, which is to speed up the migration to downstream tasks and train for 3epoch.
    # Then train normally for 7 epochs, and the learning rate is 0.0005
    optimizer = optim.Adam(params,lr=config['lr']*10)
    criterion = BPRLoss()
    import random
    max_len = config['max_len']
    train_dataset = HyperSeqDataset(dim_features,dim_features_item,max_len,train_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)

    user_seqs = []

    test_max_len = max_len -1
    for i in range(len(train_list)):
        if len(train_list[i][1:]) < test_max_len:
            user_seqs.append([train_list[i][0]]*(test_max_len-len(train_list[i][1:]))+train_list[i][1:])
        else:
            user_seqs.append(train_list[i][-test_max_len:])
    user_seqs = torch.tensor(user_seqs).to(device)

    testRatings, testNegatives = data.test_items,data.neg_items
    best_state, best_val, best_epoch = None, 0, -1
    print("train start")
    for epoch in range(1,epochs+1):
        if epoch > 3 :
            optimizer.param_groups[0]['lr'] = config['lr']

        train(net, X, hg, train_loader, optimizer, epoch, criterion)

        net_prompt.eval()
        with torch.no_grad():
            h,embs = net_prompt(u_embs,i_embs,ui_bigraph,user_seqs)
            embs = torch.cat([h+embs[:dim_features,:],embs[dim_features:,:]]).to(device)
        ret = test_united(embs, testRatings, testNegatives)
        perf_str = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
            epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
            ret['ndcg'][2])




