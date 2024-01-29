import random
import re
import shutil
import time
from itertools import combinations

import pandas as pd
import os
import pickle

import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
import json
import numpy as np
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

def preprocess():
    pat = r'(^\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*-?\d*(\.\d+)?\s*\|\s*-?\d*(\.\d+)?\s*\|\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$)'
    #

    with open('../data/foursquare/checkins.dat','r') as f:
        data = f.read()
    # print(data)
    # timedata = re.findall(pat2, data)
    data = re.findall(pat,data,flags=re.MULTILINE)
    data = [i[0].split('|') for i in data]
    # print(len(data),len(timedata))
    data = [[i[1].strip(), i[2].strip(),i[5].strip()] for i in data]
    df = pd.DataFrame(data, columns=['user', 'item', 'time'])
    df.to_csv('../data/foursquare/checkins.csv', sep='\t',index=False)

    pat = r'(^\s*\d+\s*\|\s*[-+]?\d*\.\d+\s*\|\s*[-+]?\d*\.\d+\s*$)'
    with open('../data/foursquare/users.dat') as f:
        data = f.read()
    data = re.findall(pat, data,flags=re.MULTILINE)
    # print(data)
    data = [i.split('|') for i in data]
    data = [[i[0].strip(), i[1].strip(),i[2].strip()] for i in data]
    df = pd.DataFrame(data,columns=['user','lat','lon'])
    df.to_csv('../data/foursquare/users.csv',sep='\t',index=False)

    with open('../data/foursquare/venues.dat') as f:
        data = f.read()
    data = re.findall(pat, data,flags = re.MULTILINE)
    data = [i.split('|') for i in data]
    data = [[i[0].strip(), i[1].strip(),i[2].strip()] for i in data]
    df = pd.DataFrame(data,columns=['item','lat','lon'])
    df.to_csv('../data/foursquare/venues.csv',sep='\t',index=False)

    pat = r'(\d+\s*\|\s*\d+)'
    with open('../data/foursquare/socialgraph.dat') as f:
        data = f.read()
    data = re.findall(pat, data,re.MULTILINE)
    data = [i.split('|') for i in data]
    data = [[i[0].strip(), i[1].strip()] for i in data]
    df = pd.DataFrame(data,columns=['user','friend'])

    df = df.sort_values(by=['user'])
    df.to_csv('../data/foursquare/socialgraph.csv',sep='\t',index=False,header=0)

    df = pd.read_csv('../data/foursquare/checkins.csv',sep='\t')

    df2 = pd.read_csv('../data/foursquare/venues.csv',sep='\t')

    df = pd.merge(df,df2,on='item')
    df = df[['user','time','lat','lon','item']]
    df = df.sort_values(by=['user','time'])
    df.to_csv('../data/foursquare/checkins2.csv',sep='\t',index=False,header=0)
def preprocessRating():
    df = pd.read_csv('../data/foursquare/loc-foursquare_totalCheckins_remap.txt',sep='\t',header=None,names=['user','time','ltt','att','item'])
    num_user = max(df['user'])+1
    # df.to_csv('../data/foursquare/rating.txt',sep='\t',header=None,index=None)
    df = pd.read_csv("../data/foursquare/foursquare.train.rating", sep=' ', header=None,names=['user','item','rating'])
    df = df[['user','item']]
    df['item'] += num_user
    df.to_csv('../data/foursquare/foursquare.train.rating',sep=' ',index=False,header=None)
    df = pd.read_csv("../data/foursquare/foursquare.test.rating", sep=' ', header=None,names=['user','item','rating'])
    df = df[['user','item']]
    df['item'] += num_user
    df.to_csv('../data/foursquare/foursquare.test.rating',sep=' ',index=False,header=None)

    negativedict = load_test_negative_as_dict('../data/foursquare/foursquare.test.negative')
    with open('../data/foursquare/foursquare.test.negative','w') as f:
        for user in negativedict:
            f.write(str(user)+' ')
            for item in negativedict[user]:
                f.write(str(item+num_user)+' ')
            f.write('\n')


def foursquareDR():
    df = pd.read_csv('../data/foursquare/checkins2.csv', sep='\t', header=None,
                     names=['user', 'time', 'ltt', 'att', 'item'])

    remains = pd.read_csv('../data/foursquare/remains_user_item.txt', sep=' ', header=None,names=['user','item'])
    df = df[df['user'].isin(remains['user'].tolist())]
    df = df[df['item'].isin(remains['item'].tolist())]

    remap_users = pd.read_csv('../data/foursquare/rating_ramap_user.txt', sep=' ', header=None, names=['user', 'remap_user'])
    remap_items = pd.read_csv('../data/foursquare/rating_ramap_item.txt', sep=' ', header=None, names=['item', 'remap_item'])

    remap_users = remap_users.drop_duplicates()
    remap_items = remap_items.drop_duplicates()
    df = df.merge(remap_users, on='user', how='left')
    df = df.merge(remap_items, on='item', how='left')

    df_remap = df[['remap_user', 'time', 'ltt', 'att', 'remap_item']]
    df_remap.to_csv('../data/foursquare/loc-foursquare_totalCheckins_remap.txt', sep='\t', header=None, index=None)

    df['item'] = df['remap_item'] + max(df['remap_user']) + 1
    print("max user:",max(df['remap_user']))


    df_friend = pd.read_csv('../data/foursquare/socialgraph.csv',sep='\t',header=None,names=['user1','user2'])


    df_friend['user1'] = df_friend['user1'].map(dict(zip(df['user'],df['remap_user'])))
    df_friend['user2'] = df_friend['user2'].map(dict(zip(df['user'],df['remap_user'])))

    df_friend = df_friend.dropna()

    df_friend['user1'] = df_friend['user1'].astype(int)
    df_friend['user2'] = df_friend['user2'].astype(int)

    df_friend.to_csv('../data/foursquare/loc-foursquare_edges_remap.txt',header=0,index=0,sep='\t')

    df_user = pd.read_csv('../data/foursquare/users.csv',sep='\t')
 
    df_user['user'] = df_user['user'].map(dict(zip(df['user'],df['remap_user'])))
    
    df_user = df_user.dropna()

    df_user['user'] = df_user['user'].astype(int)

    df_user.to_csv('../data/foursquare/user_remap.txt',header=0,index=0,sep='\t')



    user_hg = []
    last = -1
    userset = set()
    with open('../data/foursquare/loc-foursquare_edges_remap.txt','r') as f:
        dataset = f.readlines()
        neigh = []

        for record in dataset:
            record = record.strip('\n')
            src,dst = record.split('\t')
            src = int(src)
            dst = int(dst)

            if last != src and last != -1:
                neigh.append(last)
                user_hg.append(neigh)
                neigh = []
            neigh.append(dst)
            last = src
            userset.add(src)
        user_hg.append(neigh)

    num_vertices = max(df['remap_user'])+1
    num_hyperedges = len(user_hg)

    with open('../data/foursquare/user_hyperedge_list.pkl', 'wb') as f:
        pickle.dump(user_hg, f)

    user_info = {'num_vertices':int(num_vertices),'num_hyperedge':int(num_hyperedges)}
    with open('../data/foursquare/userInfo.json','w') as f:
        json.dump(user_info,f)
    print(user_info)




    df_unique = df.drop_duplicates(subset='item').sort_values('item')
    # 使用kmeans聚类
    cluster_model = KMeans(n_clusters=80)

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



    with open('../data/foursquare/item_hyperedge_list.pkl', 'wb') as f:
        pickle.dump(item_hg, f)


    item_info = {'num_vertices': num_item_vertices, 'num_hyperedge': num_item_hyperedges}
    with open('../data/foursquare/itemInfo.json', 'w') as f:
        json.dump(item_info, f)


    df['user'] = df['remap_user']



    df_user = pd.read_csv('../data/foursquare/user_remap.txt', sep='\t', header=None, names=['user', 'lon', 'lat'])

    df_user = df_user.groupby(['lon','lat'])['user'].apply(list).reset_index()

    users_neighbors = []
    for i in range(len(df_user)):
        user_neighbors = df_user['user'][i]
        users_neighbors.append(user_neighbors)


    with open('../data/foursquare/user_neighbors.pkl','wb') as f:
        pickle.dump(users_neighbors,f)

    user_neighbors_info = {'num_vertices':len(df_user),'num_hyperedge':len(users_neighbors)}
    with open('../data/foursquare/user_neighbors_info.json','w') as f:
        json.dump(user_neighbors_info,f)

def foursquareDR2():
    df_user = pd.read_csv('../data/foursquare/user_remap.txt', sep='\t', header=None, names=['user', 'lon', 'lat'])

    df_user = df_user.groupby(['lon', 'lat'])['user'].apply(list).reset_index()

    users_neighbors = []
    for i in range(len(df_user)):
        user_neighbors = df_user['user'][i]
        users_neighbors.append(user_neighbors)

    with open('../data/foursquare/user_neighbors.pkl', 'wb') as f:
        pickle.dump(users_neighbors, f)

    user_neighbors_info = {'num_vertices': len(df_user), 'num_hyperedge': len(users_neighbors)}
    with open('../data/foursquare/user_neighbors_info.json', 'w') as f:
        json.dump(user_neighbors_info, f)



def foursquareSeq():
    # 第一步先调用foursquareDR函数来生成对应的数据集
    # 读取/data/foursquare/loc-foursquare_totalCheckins_remap.txt文件
    df = pd.read_csv('../data/foursquare/loc-foursquare_totalCheckins_remap.txt', sep='\t', header=None,
                     names=['user', 'time', 'ltt', 'att', 'item'])
    # 统计user的数量和item的数量
    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1
    # 将item加上user_num，使得item的id不会和user的id重复
    df['item'] += user_num
    df = df.groupby('user').apply(lambda x: x.sort_values('time'))
    # 还原df
    # q: 为什么？
    # a: 因为groupby之后，df的index会变成user，所以需要reset_index
    df = df.reset_index(drop=True)
    # 生成train_dict
    train_dict = {}
    for user in df['user'].unique():
        train_dict[user] = df[df['user'] == user]['item'].tolist()[:-1]
    # 生成test_dict
    test_dict = {}
    for user in df['user'].unique():
        test_dict[user] = df[df['user'] == user]['item'].tolist()[-1]
    # 生成负样本neg_dict，负样本数量为99
    neg_dict = {}
    for user in df['user'].unique():
        neg_dict[user] = []
        for i in range(99):
            neg_sample = random.randint(user_num, user_num+item_num - 1)
            while neg_sample == test_dict[user]:
                neg_sample = random.randint(user_num, user_num+item_num - 1)
            neg_dict[user].append(neg_sample)
    # 写出train_dict
    with open('../data/foursquare/foursquare.seq.train.seq', 'w') as f:
        for user in train_dict.keys():
            f.write(str(user) + ' ' + ' '.join([str(i) for i in train_dict[user]]) + '\n')
    # 写出test_dict
    with open('../data/foursquare/foursquare.seq.test.target', 'w') as f:
        for user in test_dict.keys():
            f.write(str(user) + ' ' + str(test_dict[user]) + '\n')
    # 写出neg_dict
    with open('../data/foursquare/foursquare.seq.test.negative', 'w') as f:
        for user in neg_dict.keys():
            f.write(str(user)+ ' ' + " ".join([str(i) for i in neg_dict[user]]) + '\n')








if __name__ == '__main__':
    # foursquareSeqForSTAN()
    foursquareDR()

