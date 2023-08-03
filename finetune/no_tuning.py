import sys

import sklearn

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
from data.dataset import Gowalla, Foursquare, Amazon
from models.HGNNP import HGNNP, HGNNP_NoLight
from dhg.random import set_seed
from dhg.experiments import HypergraphVertexClassificationTask as Task
from dhg.metrics import UserItemRecommenderEvaluator as Evaluator
from dhg.nn import BPRLoss, EmbeddingRegularization
import logging
import os
from torch.utils.data import DataLoader
from utils.batch_test import test_united
import random


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
    # Parse the hyperparameter.
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gowalla')
    parser.add_argument('--pret_mode',type=str,default='LP_PM')
    parser.add_argument("--pretrain_model", type=str, default="500")
    args = parser.parse_args()
    print("==================================================")
    print("pretrain mode:",args.pret_mode)

    with open('../config/finetune.json','r') as f:
        config = json.load(f)
    set_seed(config['seed'])
    seed_torch(config['seed'])
    dim_emb = config['dim_emb']
    batch_sz = config['batch_sz']

    epochs = config['epochs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_freq = config['val_freq']

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s]','%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    curtime = datetime.date.today()
    logfile = logging.FileHandler(os.path.join(work_root+'exp_cache/',args.dataset+"-"+args.pretrain_model,str(curtime)+"notuning_log.txt"))
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    # evaluator = Evaluator([{"ndcg": {"k": 20}}, {"recall": {"k": 20}}])
    if args.dataset == 'gowalla':
        data = Gowalla(data_root=work_root+'data/')
    elif args.dataset == 'foursquare':
        data = Foursquare(data_root=work_root+'data/')
    elif args.dataset == 'amazon':
        data = Amazon(data_root=work_root+'data/')
    else:
        raise ValueError("dataset must be gowalla or foursquare or amazon")
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

    pretrained_dict = torch.load('../save/model/'+args.dataset+f'/HGNNP_plus_LP_PM_{args.pretrain_model}.pth')
    model_dict = net.state_dict()
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    gcn_train_list = [[user,item-num_u] for user,item in train_list]
    ui_bigraph = BiGraph.from_adj_list(num_u, num_i, gcn_train_list)
    ui_bigraph = ui_bigraph.to(device)

    train_edge_list = adj_list_to_edge_list(train_list)
    test_edge_list = adj_list_to_edge_list(test_list)

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

    net.eval()
    with torch.no_grad():
        nodes_emb, hedges_emb = net(X, hg)
    nodes_emb = torch.nn.functional.normalize(nodes_emb, dim=-1).detach()
    hedges4node_emb = torch.nn.functional.normalize(torch.matmul(node_hedge_inci_matrix,
                                                                 hedges_emb), dim=-1).detach()
    embs = nodes_emb + hedges4node_emb

    testRatings, testNegatives = data.test_items,data.neg_items

    ret = test_united(embs, testRatings, testNegatives)
    perf_str = 'Dataset:%s Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
        args.dataset,
        0, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
        ret['ndcg'][2])
    logger.info(perf_str)



