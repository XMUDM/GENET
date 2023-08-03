# encoding: utf-8

import numpy as np
import pandas as pd
import pickle
import json
import gzip
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings("ignore")



def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    check_step = 1e5
    df = {}
    print("Start reading data from {}".format(path))
    for d in parse(path):
        if i % check_step == 0:
            print("Processing {}th line ".format(i))
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def clean_text(x):
    """
    Remove special characters, html tags and notations from the given text.
    :param x:
    :return:
    """
    if type(x) == str:
        # remove htmls
        x = re.sub(r'<.*?>', '', x)
        # remove special characters
        x = re.sub(r'[^a-zA-z0-9\s]', '', x)
        # remove notations
        x = re.sub(r'@\w+', '', x)

    return x


def remap_id(data: pd.DataFrame, meta: pd.DataFrame)->pd.DataFrame:
    """
    Remap user id and item id to continuous id
    :param df: input dataframe
    :return: remapped dataframe
    """
    # remap user id
    user_id = data.reviewerID.unique()
    user_id_map = dict(zip(user_id, range(len(user_id))))
    data.reviewerID = data.reviewerID.map(user_id_map)
    # remap item id
    item_id = data.asin.unique()
    item_id_map = dict(zip(item_id, range(len(item_id))))
    data_asin = set(data.asin.unique())
    meta_asin = set(meta.asin.unique())
    inter_asin = data_asin.intersection(meta_asin)
    print("There are {} items in data and {} items in meta data, {} items in common".format(len(data_asin), len(meta_asin), len(inter_asin)))
    # map also view and also buy id to item id
    item_start = max(data['reviewerID']) + 1
    meta['also_view'] = meta['also_view'].map(lambda x: [item_id_map[i] + item_start for i in x if i in inter_asin] if x is not np.nan else [])
    meta['also_buy'] = meta['also_buy'].map(lambda x: [item_id_map[i] + item_start for i in x if i in inter_asin] if x is not np.nan else [])
    data.asin = data.asin.map(item_id_map) + item_start
    meta.asin = meta.asin.map(item_id_map) + item_start
    data, meta = load_remap("Dataset/Amazon/Books/", data, meta) # load remap file and
    num_user = len(data.reviewerID.unique())
    num_item = len(data.asin.unique())
    return data, meta, num_user, num_item


def load_remap(path: str, data:pd.DataFrame, meta:pd.DataFrame):
    """
    load remap file
    :param path: path to remap file
    :return:
    """
    user_map = {}
    item_map = {}
    user_path = path+"rating_ramap_user.txt"
    item_path = path+"rating_ramap_item.txt"

    with open(user_path, 'r') as f:
        for line in f:
            (key, val) = line.split()
            user_map[int(key)] = int(val)
    with open(item_path, 'r') as f:
        for line in f:
            (key, val) = line.split()
            item_map[int(key)] = int(val)
    # remap user and item id
    data.drop(data[~data.asin.isin(item_map.keys())].index, inplace=True)
    data.drop(data[~data.reviewerID.isin(user_map.keys())].index, inplace=True)
    meta.drop(meta[~meta.asin.isin(item_map.keys())].index, inplace=True)
    data.reviewerID = data.reviewerID.map(user_map)
    data.asin = data.asin.map(item_map)
    meta.asin = meta.asin.map(item_map)
    data = data.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    return data, meta


def compute_sentiment_score(df: pd.DataFrame):
    """
    Compute sentiment score for each review
    :param df:
    :return:
    """

    def f(x, threshold):
        if type(x) == str and  len(x.split( )) > threshold:
            return 'positive'
        else:
            return 'other'
    grouped_data = df.groupby('reviewerID')
    def review_len(x):
        if type(x) == str:
            return len(x.split())
        else:
            return 0
    for user, group in grouped_data:
        threshold = group['reviewText'].map(lambda x: review_len(x)).quantile(0.9)
        group['sentiment'] = group['reviewText'].map(lambda x: f(x, threshold) if type(x) == str else 'other')
        df.loc[group.index, 'sentiment'] = group['sentiment']

    return df



def build_item_hypergraph(data, meta):
    """
    Build item hypergraph from data and meta data using category and brand information
    :param data: dataset
    :param meta: meta data
    :return: item hypergraph
    """

    print("Building category dictionary")
    category2id = {}
    for i in tqdm(range(len(meta.category))):
        for j in range(len(meta.category[i])):
            if meta.category[i][j] not in category2id:
                category2id[meta.category[i][j]] = len(category2id)
    num_category = len(category2id)
    print("category id range: {}-{}".format(min(category2id.values()), max(category2id.values())))
    # mapping brand, category
    data.brand = data.brand.map(lambda x: x if x is not np.nan else 'Unknown')
    num_brand = len(meta.brand.unique())
    brand2id = dict(zip(data.brand.unique(), range(num_brand)))
    print("brand id range: {}-{}".format(min(brand2id.values()), max(brand2id.values())))
    data.brand = data.brand.map(brand2id)
    meta.brand = meta.brand.map(brand2id)
    data['category'] = data['category'].map(lambda x: x if (x is not np.nan and x != []) else [])
    data['category'] = data['category'].map(lambda x: [category2id[i] for i in x])
    meta['category'] = meta['category'].map(lambda x: [category2id[i] for i in x])
    meta = meta[meta['description'].map(lambda d: len(d)) > 0]
    meta = meta[meta['title'].map(lambda d: len(d)) > 0]
    data['reviewText'] = data.reviewText.apply(clean_text)
    meta['description'] = meta.description.apply(lambda x: [clean_text(i) for i in x])
    meta['title'] = meta.title.apply(clean_text)

    asin2id = {}
    id2asin = {}
    for i, asin in enumerate(data.asin.unique()):
        asin2id[asin] = i
        id2asin[i] = asin
    brand_hg = [[] for _ in range(num_brand)]
    # build hypergraph
    for i, row in tqdm(meta.iterrows()):
        asin = row.asin
        brand = row.brand
        brand_hg[brand].append(asin)

    category_hg = [[] for _ in range(num_category)]
    for i, row in tqdm(meta.iterrows()):
        asin = row.asin
        for j in row.category[1:]:
            category_hg[j].append(asin)
    print("item hypergraph constructed")

    return brand_hg, category_hg




def build_user_hypergraph(data: pd.DataFrame, meta: pd.DataFrame, threshold:float=0.9)->list:
    """

    if a user shows positive sentiment on a certain brand, then he/she will be more likely to buy items from that brand
    :param data: dataset
    :param meta: meta data
    :param threshold: threshold to decide whether a user shows positive sentiment on a certain brand
    :return:
    """
    print("Constructing user hypergraph...")
    category2id = {}
    for i in tqdm(range(len(meta.category))):
        for j in range(len(meta.category[i])):
            if meta.category[i][j] not in category2id:
                category2id[meta.category[i][j]] = len(category2id)
    num_category = len(category2id)

    data.brand = data.brand.map(lambda x: x if x is not np.nan else 'Unknown')
    num_brand = len(meta.brand.unique())
    brand2id = dict(zip(data.brand.unique(), range(num_category, num_category + num_brand)))
    data.brand = data.brand.map(brand2id)
    meta.brand = meta.brand.map(brand2id)
    data['category'] = data.category.map(lambda x: [category2id[i] for i in x])
    meta['category'] = meta.category.map(lambda x: [category2id[i] for i in x])

    user_hyperedge = [[] for i in range(num_category)]
    data = compute_sentiment_score(data)
    grouped_data = data.groupby('reviewerID')
    for user, group in grouped_data:
        record = [{'positive': 0, 'total': 0} for i in range(num_category)]
        for _, row in group.iterrows():
            if row['sentiment'] == 'positive':
                for c in row['category'][1:]:
                    record[c]['positive'] += 1
            for c in row['category'][1:]:
                record[c]['total'] += 1
        for i in range(num_category):
            if record[i]['total'] > 0 and record[i]['positive'] / record[i]['total'] > threshold:
                user_hyperedge[i].append(user)
    print("User hypergraph constructed")

    return user_hyperedge








def build_hypergraph(data_name: str, data_path: str, meta_path: str):
    """
    Build hypergraphs for users and items
    :param data_name: name of the dataset
    :param data_path: path of the dataset
    :param meta_path: path of the meta data
    :return:
    A hyperedge set for both users and items
    """
    data = getDF(data_path)
    meta = getDF(meta_path)
    data, meta, num_user, num_items = remap_id(data, meta)
    data = pd.merge(data, meta[['asin', 'category', 'brand']], on=['asin'], how='left')
    brand_hg, category_hg = build_item_hypergraph(data, meta)
    user_hg = build_user_hypergraph(data, meta, threshold=0.9)
    return user_hg, brand_hg, category_hg




if __name__ == "__main__":
    data_name = "Books"
    data_path = "Dataset/{}_5.json.gz".format(data_name)
    meta_path = "Dataset/meta_{}.json.gz".format(data_name)
    user_hg, brand_hg, category_hg = build_hypergraph(data_name, data_path, meta_path)
    save_dir = "Dataset/".format(data_name)
    def remap_item_hg(item_hg, num_user):
        num_item = 0
        new_item_hg = []
        for i in range(len(item_hg)):
            if len(item_hg[i]) > 0 and num_item < max(item_hg[i]):
                num_item = max(item_hg[i])
            for j in range(len(item_hg[i])):
                item_hg[i][j] += num_user
            new_item_hg.append(item_hg[i])

        num_item += 1
        # remove empty hyperedges
        new_item_hg = [i for i in new_item_hg if len(i) > 0]
        item_info = {'num_vertices': int(num_item), 'num_hyperedges': int(len(new_item_hg))}
        return new_item_hg, item_info, num_item


    # unify the id of users and items
    num_user = 0
    for i in range(len(user_hg)):
        if len(user_hg[i]) > 0 and num_user < max(user_hg[i]):
            num_user = max(user_hg[i])

    user_hg = [i for i in user_hg if len(i) > 0]
    num_user += 1
    user_info = {'num_vertices': int(num_user), 'num_hyperedges': int(len(user_hg))}
    brand_item_hg, brand_item_info, brand_num_item = remap_item_hg(brand_hg, num_user)
    category_item_hg, category_item_info, category_num_item = remap_item_hg(category_hg, num_user)

    with open('{}/Books_user_hyperedge.pkl'.format(save_dir), 'wb') as f:
        pickle.dump(user_hg, f)
        f.close()

    with open('{}/Books_brand_item_hyperedge.pkl'.format(save_dir), 'wb') as f:
        pickle.dump(brand_item_hg, f)
        f.close()

    with open('{}/Books_category_item_hyperedge.pkl'.format(save_dir), 'wb') as f:
        pickle.dump(category_item_hg, f)
        f.close()


    with open('{}/Books_user_info.json'.format(save_dir), 'w') as f:
        json.dump(user_info, f)

    with open('{}/Books_brand_item_info.json'.format(save_dir), 'w') as f:
        json.dump(brand_item_info, f)

    with open('{}/Books_category_item_info.json'.format(save_dir), 'w') as f:
        json.dump(category_item_info, f)

