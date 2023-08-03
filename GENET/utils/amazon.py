import random
import re
import shutil
import time

import pandas as pd
import os
import pickle
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

def amazonSeq():
    df = pd.read_csv('../data/amazon/book_ratings_seq.txt', sep='\t', header=None,
                     names=['time','rating','user', 'item'])
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
    with open('../data/amazon/amazon.seq.train.seq', 'w') as f:
        for user in train_dict.keys():
            f.write(str(user) + ' ' + ' '.join([str(i) for i in train_dict[user]]) + '\n')
    with open('../data/amazon/amazon.seq.test.target', 'w') as f:
        for user in test_dict.keys():
            f.write(str(user) + ' ' + str(test_dict[user]) + '\n')
    with open('../data/amazon/amazon.seq.test.negative', 'w') as f:
        for user in neg_dict.keys():
            f.write(str(user)+ ' ' + " ".join([str(i) for i in neg_dict[user]]) + '\n')



def preprocessRatingSeq():
    df = pd.read_csv('../data/amazon/book_ratings.txt',header=None,sep='\t',names=['time','rating','user','item',])
    num_u = max(df['user']) + 1
    df['item'] = df['item'] - num_u
    df = df.sort_values(by=['user','time'])
    # 每一个user只保留最后30条记录
    df = df.groupby('user').tail(30)
    df.to_csv('../data/amazon/book_ratings_seq.txt',header=None,index=False,sep='\t')


if __name__ == '__main__':
    amazonSeq()
