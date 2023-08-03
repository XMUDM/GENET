import os
import pickle
import shutil

import numpy as np
import pandas as pd

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

def build_cold_start_dataset(trainfn,testfn,negfn,dataset):
    traindf = pd.read_csv(trainfn,header=None,names=['user','item'],sep=' ')
    testdict = load_test_rating_as_dict(testfn)
    negdict = load_test_negative_as_dict(negfn)

    user_item_count = traindf.groupby('user')['item'].count()

    item_user_count = traindf.groupby('item')['user'].count()


    item_user_count = item_user_count[item_user_count <= item_user_count.quantile(0.01)]

    user_item_count = user_item_count[user_item_count <= user_item_count.quantile(0.01)]

    traindf_item_coldstart = traindf[~traindf['item'].isin(item_user_count.index)]

    traindf_user_coldstart = traindf[~traindf['user'].isin(user_item_count.index)]

    testdict_user_coldstart = {k: v for k, v in testdict.items() if k in user_item_count.index}

    negdict_user_coldstart = {k: v for k, v in negdict.items() if k in user_item_count.index}


    testdict_item_coldstart = {}

    negdict_item_coldstart = {}
    for k,v in testdict.items():
        if v[0] in item_user_count.index:
            testdict_item_coldstart[k] = v
            negdict_item_coldstart[k] = negdict[k]


    traindf_item_coldstart.to_csv(os.path.join(os.path.dirname(trainfn),f'{dataset}.train.ics.rating'),header=False,index=False,sep=' ')
    traindf_user_coldstart.to_csv(os.path.join(os.path.dirname(trainfn),f'{dataset}.train.ucs.rating'),header=False,index=False,sep=' ')
    with open(os.path.join(os.path.dirname(testfn),f'{dataset}.test.ics.rating'),'w') as f:
        for k,v in testdict_item_coldstart.items():
            f.write(f'{k} {v[0]}\n')
    with open(os.path.join(os.path.dirname(testfn),f'{dataset}.test.ucs.rating'),'w') as f:
        for k,v in testdict_user_coldstart.items():
            f.write(f'{k} {v[0]}\n')
    with open(os.path.join(os.path.dirname(negfn),f'{dataset}.test.ics.negative'),'w') as f:
        for k,v in negdict_item_coldstart.items():
            f.write(f'{k} {" ".join([str(x) for x in v])}\n')
    with open(os.path.join(os.path.dirname(negfn),f'{dataset}.test.ucs.negative'),'w') as f:
        for k,v in negdict_user_coldstart.items():
            f.write(f'{k} {" ".join([str(x) for x in v])}\n')

if __name__ == '__main__':
    build_cold_start_dataset(
        trainfn='../data/gowalla/gowalla.train.rating',
        testfn='../data/gowalla/gowalla.test.rating',
        negfn='../data/gowalla/gowalla.test.negative',
        dataset='gowalla'
    )




