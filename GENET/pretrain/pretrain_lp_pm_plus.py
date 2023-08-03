import sys
sys.path.append("..")
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import datetime

from dhg import Hypergraph
from dhg.utils import UserItemDataset, sparse_dropout
import torch.nn.functional as F
from data.dataset_wrapers import HyperDataset
from data.dataset import Gowalla, Foursquare, Amazon
from models.HGNNP import HGNNP
from dhg.random import set_seed
from dhg.experiments import HypergraphVertexClassificationTask as Task
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.nn import BPRLoss, EmbeddingRegularization
import logging
import os
from torch.utils.data import DataLoader
import time
import argparse


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

def train(net,X,A,data_loader,optimizer,epoch,criterion):
    net.train()
    loss_mean = 0
    for hedge,anchor, pos, neg,hedge_items in data_loader:
        nodes_emb, hedges_emb = net(X, A)

        # CL loss
        A_hat = A.clone()
        A_hat = A_hat.drop_hyperedges(cl_coef)
        nodes_emb_hat, hedges_emb_hat = net(X, A_hat)
        nodes_emb_norm, hedges_emb_norm = F.normalize(nodes_emb,dim=-1), F.normalize(hedges_emb,dim=-1)
        nodes_emb_hat_norm, hedges_emb_hat_norm = F.normalize(nodes_emb_hat,dim=-1), F.normalize(hedges_emb_hat,dim=-1)
        pos_u = torch.exp(torch.sum(nodes_emb_norm[anchor] * nodes_emb_hat_norm[anchor],dim=-1) / gamma)# [batch_size]
        neg_u = torch.sum(torch.exp(torch.matmul(nodes_emb_norm[anchor],nodes_emb_norm[anchor].t())-torch.eye(anchor.size()[0]).to(device) / gamma),dim=-1) # [batch_size]
        loss_con_inter = -torch.mean(torch.log(pos_u / (pos_u+neg_u)))
        hedge_items = [hedge_items[i].tolist() for i in range(len(hedge_items))]
        pos_u_intra = torch.exp(
            torch.sum(nodes_emb_norm[anchor] * nodes_emb_hat_norm[anchor], dim=-1) / gamma)  # [batch_size]
        neg_u_intra = torch.sum(torch.exp(
            torch.matmul(nodes_emb_norm[anchor], nodes_emb_norm[hedge_items].transpose(2,1)) / gamma), dim=-1)  # [batch_size]
        loss_con_intra = -torch.mean(torch.log(pos_u_intra / (pos_u_intra+neg_u_intra)))

        # Enhance connection loss
        nodes_emb_noise = nodes_emb+torch.randn_like(nodes_emb)*ec_coef
        nodes_emb = torch.nn.functional.normalize(nodes_emb, dim=-1)
        nodes_emb_noise = torch.nn.functional.normalize(nodes_emb_noise, dim=-1)
        hedges4node_emb = torch.nn.functional.normalize(torch.matmul(A.H, hedges_emb), dim=-1)
        # H_hat = A.clone().H
        # H_hat = H_hat.coalesce()
        # del_indice = torch.stack([pos.to(device), hedge.to(device)], dim=1).\
        #     unsqueeze(1).expand(-1,H_hat.indices().t().size(0), -1)
        # mask = torch.any(torch.all(torch.eq(H_hat.indices().t(), del_indice), dim=2), dim=0)
        # H_hat.values()[mask] = 0
        # H_hat = H_hat.coalesce()
        H_hat = A.H.coalesce()
        del_indices = torch.stack([pos.to(device), hedge.to(device)], dim=1)
        # mask = torch.any(torch.all(torch.eq(H_hat.indices().t(), del_indices.unsqueeze(1)), dim=2), dim=0)
        H_hat_indices = H_hat.indices().t().unsqueeze(0)
        mask = torch.isin(H_hat_indices, del_indices.unsqueeze(0)).all(dim=2).any(dim=0)
        H_hat = torch.sparse_coo_tensor(H_hat.indices()[:, ~mask], H_hat.values()[~mask], H_hat.size())
        H_hat = H_hat.coalesce()
        hedges4node_emb_noise = torch.nn.functional.normalize(torch.matmul(H_hat, hedges_emb), dim=-1)
        nodes_emb = nodes_emb+hedges4node_emb
        nodes_emb_noise = nodes_emb_noise+hedges4node_emb_noise
        hedges_emb = torch.nn.functional.normalize(hedges_emb, dim=-1)

        anchor_emb = nodes_emb[anchor]+hedges_emb[hedge]
        # anchor_emb_norm = torch.nn.functional.normalize(anchor_emb,dim=-1)
        pos_emb = nodes_emb_noise[pos]+hedges_emb[hedge]
        neg_emb = nodes_emb[neg]+hedges_emb[hedge]
        pos_scores, neg_scores = (anchor_emb * pos_emb).sum(dim=1), (anchor_emb * neg_emb).sum(dim=1)
        optimizer.zero_grad()
        loss1 = criterion(pos_scores,neg_scores)

        loss  = loss1 + intra_weight*loss_con_intra+inter_weight*loss_con_inter

        print(loss1.item(),loss_con_intra.item(),loss_con_inter.item())
        loss.backward()
        optimizer.step()
        loss_mean += loss.item()
    loss_mean /= len(data_loader)
    logger.info(f"Epoch:{epoch}, Loss:{loss_mean:.5f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='amazon')
    args = parser.parse_args()
    work_root = "your work root"
    assert work_root != "your work root", "Please set your work root"
    with open(os.path.join(work_root,'config/pretrain.json'),'r') as f:
        config = json.load(f)
    set_seed(config['seed'])
    seed_torch(config['seed'])
    dim_emb = config['dim_emb']
    gamma = config['cl_gamma']
    batch_sz = config['batch_sz']
    cl_coef = config['cl_coef']
    ec_coef = config['ec_coef']
    epochs = config['epochs']
    intra_weight = config['intra_weight']
    inter_weight = config['inter_weight']
    device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s]','%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    curtime = datetime.date.today()
    if not os.path.exists(os.path.join(work_root,'exp_cache',args.dataset+"-"+str(epochs))):
        os.mkdir(os.path.join(work_root,'exp_cache',args.dataset+"-"+str(epochs)))
    logfile = logging.FileHandler(os.path.join(os.path.join(work_root,'exp_cache',args.dataset+"-"+str(epochs),str(curtime)+"_pretrain_plus_log.txt")))
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    if args.dataset == 'gowalla':
        data = Gowalla()
    elif args.dataset == 'foursquare':
        data = Foursquare()
    elif args.dataset == 'amazon':
        data = Amazon()
    else:
        raise ValueError("dataset must be gowalla or foursquare or amazon")
    dim_features = data["num_user_vertices"]
    dim_features_item = data["num_item_vertices"]
    if args.dataset == 'foursquare':
        hg = Hypergraph(data["num_user_vertices"]+data['num_item_vertices'],
                    data["user_edge_list"]+data['item_edge_list']+data['user_neighbors']).to(device)
    elif args.dataset == 'gowalla':
        hg = Hypergraph(data["num_user_vertices"] + data['num_item_vertices'],
                        data["user_edge_list"] + data['item_edge_list']).to(device)
    elif args.dataset == 'amazon':
        hg = Hypergraph(data["num_user_vertices"]+data['num_item_vertices'],
                        data["user_edge_list"]+data['item_edge_list']+data['item_edge_list2']).to(device)
    else:
        raise ValueError("dataset must be gowalla or foursquare or amazon")


    X = torch.sparse_coo_tensor(torch.arange(dim_features+dim_features_item).unsqueeze(0).repeat(2,1),
                                torch.ones(dim_features+dim_features_item),
                                torch.Size([dim_features+dim_features_item,dim_features+dim_features_item])).to(device)

    net = HGNNP(X.shape[1],dim_emb,use_bn=True).to(device)
    data_loader = 0
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    criterion = BPRLoss()
    train_dataset = HyperDataset(dim_features,dim_features_item,hg)
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    logger.info(f"hypergraph pretraining.")

    # save
    if not os.path.exists(os.path.join(work_root,"save/structure/",args.dataset)):
        os.makedirs(os.path.join(work_root,"save/structure/",args.dataset))
    if not os.path.exists(os.path.join(work_root,"save/model/",args.dataset)):
        os.makedirs(os.path.join(work_root,"save/model/",args.dataset))
    # save the hypergraph
    hg.save(os.path.join(work_root,"save/structure/",args.dataset,"hg_plus.pkl"))
    for epoch in range(epochs):
        train(net,X,hg,train_loader,optimizer,epoch,criterion)
    # save the pretrained model
    torch.save(net.state_dict(),os.path.join(work_root,'save/model/',args.dataset,f'HGNNP_plus_LP_PM_{epochs}.pth'))





