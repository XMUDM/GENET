
import gc
import math
import shutil
import time

import scipy.sparse as sp
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import pickle
import json
import dhg
import math
import pandas as pd
import numpy as np
import os
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
import random

def load_test_rating_as_dict(filename):
    ratingdict = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                arr = line.split(" ")  # \t
                user, item = int(arr[0]), int(arr[1])
                ratingdict[user] = [item]
    return ratingdict
def load_test_negative_as_dict(filename):
    negativedict = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                arr = line.split(" ")
                user = int(arr[0])
                negatives = []
                for x in arr[1:]:
                    x = x.strip()
                    if x == "":
                        continue
                    negatives.append(int(x))
                negativedict[user] = negatives
    return negativedict

def gowallaSeq():

    df = pd.read_csv('../data/gowalla/loc-gowalla_totalCheckins_remap.txt', sep='\t', header=None,
                     names=['user', 'time', 'ltt', 'att', 'item'])

    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1
    df['item'] += user_num
    df = df.groupby('user').apply(lambda x: x.sort_values('time'))

    df = df.reset_index(drop=True)

    train_dict = {}
    for user in df['user'].unique():
        train_dict[user] = df[df['user'] == user]['item'].tolist()[:-1]

    test_dict = {}
    for user in df['user'].unique():
        test_dict[user] = df[df['user'] == user]['item'].tolist()[-1]

    neg_dict = {}
    for user in df['user'].unique():
        neg_dict[user] = []
        for i in range(99):
            neg_sample = random.randint(user_num, user_num+item_num - 1)
            while neg_sample == test_dict[user]:
                neg_sample = random.randint(user_num, user_num+item_num - 1)
            neg_dict[user].append(neg_sample)

    with open('../data/gowalla/gowalla.seq.train.seq', 'w') as f:
        for user in train_dict.keys():
            f.write(str(user) + ' ' + ' '.join([str(i) for i in train_dict[user]]) + '\n')

    with open('../data/gowalla/gowalla.seq.test.target', 'w') as f:
        for user in test_dict.keys():
            f.write(str(user) + ' ' + str(test_dict[user]) + '\n')

    with open('../data/gowalla/gowalla.seq.test.negative', 'w') as f:
        for user in neg_dict.keys():
            f.write(str(user)+ ' ' + " ".join([str(i) for i in neg_dict[user]]) + '\n')

def gowallaDR():
    df = pd.read_csv('../data/gowalla/loc-gowalla_totalCheckins.txt', sep='\t', header=None,
                     names=['user', 'time', 'ltt', 'att', 'item'])

    remains = pd.read_csv('../data/gowalla/remains_user_item.txt', sep=' ', header=None,names=['user','item'])
    df = df[df['user'].isin(remains['user'].tolist())]
    df = df[df['item'].isin(remains['item'].tolist())]

    remap_users = pd.read_csv('../data/gowalla/remap_users.txt', sep=' ', header=None, names=['user', 'remap_user'])
    remap_items = pd.read_csv('../data/gowalla/remap_items.txt', sep=' ', header=None, names=['item', 'remap_item'])

    remap_users = remap_users.drop_duplicates()
    remap_items = remap_items.drop_duplicates()
    df = df.merge(remap_users, on='user', how='left')
    df = df.merge(remap_items, on='item', how='left')

    df_remap = df[['remap_user', 'time', 'ltt', 'att', 'remap_item']]
    df_remap.to_csv('../data/gowalla/loc-gowalla_totalCheckins_remap.txt', sep='\t', header=None, index=None)

    df['item'] = df['remap_item'] + max(df['remap_user']) + 1
    print("max user:",max(df['remap_user']))

    df_friend = pd.read_csv('../data/gowalla/loc-gowalla_edges.txt',sep='\t',header=None,names=['user1','user2'])


    df_friend['user1'] = df_friend['user1'].map(dict(zip(df['user'],df['remap_user'])))
    df_friend['user2'] = df_friend['user2'].map(dict(zip(df['user'],df['remap_user'])))

    df_friend = df_friend.dropna()

    df_friend['user1'] = df_friend['user1'].astype(int)
    df_friend['user2'] = df_friend['user2'].astype(int)

    df_friend.to_csv('../data/gowalla/loc-gowalla_edges_remap.txt',header=0,index=0,sep='\t')

    user_hg = []
    last = -1
    userset = set()
    with open('../data/gowalla/loc-gowalla_edges_remap.txt','r') as f:
        dataset = f.readlines()
        neigh = []

        for record in dataset:
            record = record.strip('\n')
            src,dst = record.split('\t')
            src = int(src)
            dst = int(dst)

            if last != src and last != -1:
                user_hg.append(neigh)
                neigh = []
            neigh.append(dst)
            last = src
            userset.add(src)
        user_hg.append(neigh)

    num_vertices = max(df['remap_user'])+1
    num_hyperedges = len(user_hg)

    with open('../data/gowalla/user_hyperedge_list.pkl', 'wb') as f:
        pickle.dump(user_hg, f)

    user_info = {'num_vertices':int(num_vertices),'num_hyperedge':int(num_hyperedges)}
    with open('../data/gowalla/userInfo.json','w') as f:
        json.dump(user_info,f)
    print(user_info)


    df_unique = df.drop_duplicates(subset='item').sort_values('item')

    cluster_model = KMeans(n_clusters=100)

    print(df_unique.shape)
    df_unique = df_unique.dropna()
    print(df_unique.shape)
    cluster_model.fit(df_unique[['ltt','att']])
    df_unique['cluster'] = cluster_model.labels_

    item_hg = []
    for cluster in df_unique['cluster'].unique():
        item_hg.append(df_unique[df_unique['cluster']==cluster]['item'].values.tolist())

    num_item_hyperedges = len(item_hg)
    num_item_vertices = int(df_unique['item'].max()-df_unique['item'].min()+1)
    print(num_item_hyperedges,num_item_vertices)



    with open('../data/gowalla/item_hyperedge_list.pkl', 'wb') as f:
        pickle.dump(item_hg, f)


    item_info = {'num_vertices': num_item_vertices, 'num_hyperedge': num_item_hyperedges}
    with open('../data/gowalla/itemInfo.json', 'w') as f:
        json.dump(item_info, f)


    df['user'] = df['remap_user']


    user_item_list = []
    for user in df['remap_user'].unique():
        ui_list = df[df['remap_user'] == user]['item'].values.tolist()
        ui_list.append(user)
        user_item_list.append(ui_list)
    print(user_item_list[0])

    with open('../data/gowalla/user_item_hyperedge_list.pkl', 'wb') as f:
        pickle.dump(user_item_list, f)

    user_item_info = {'num_vertices': num_vertices, 'num_hyperedge': len(user_item_list)}
    with open('../data/gowalla/userItemInfo.json', 'w') as f:
        json.dump(user_item_info, f)



if __name__ == '__main__':
    # gowallaSeqForGRU4REC()
    # gowallaSeqForFlashback()
    # gowallaDGCF()
    # gowallaSeqForS3()
    gowallaDR ()
