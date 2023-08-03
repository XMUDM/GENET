import hashlib

from dhg.data import BaseData
import json
import os
from dhg.datapipe import load_from_pickle, load_from_txt
from hyperopt import partial
import pandas as pd

from utils.gowalla import load_test_negative_as_dict, load_test_rating_as_dict

class Gowalla(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("gowalla",data_root)
        with open(data_root+'gowalla/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'gowalla/itemInfo.json', 'r') as f:
            item_config = json.load(f)
        with open(data_root+'gowalla/userItemInfo.json', 'r') as f:
            user_item_config = json.load(f)

        with open(data_root+'gowalla/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'gowalla/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'gowalla/user_item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/gowalla.test.rating', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/gowalla.train.rating', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/testFriendEdges.txt', 'rb') as f:
            data = f.read()
        fri_test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/trainFriendEdges.txt', 'rb') as f:
            data = f.read()
        fri_train_list_md5 = hashlib.md5(data).hexdigest()

        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        # self.train_
        self.test_items = load_test_rating_as_dict(data_root+'gowalla/gowalla.test.rating')

        self.neg_items = load_test_negative_as_dict(data_root+'gowalla/gowalla.test.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_item_vertices": user_item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "num_user_item_edges": user_item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "user_item_edge_list": {
                "upon": [
                    {
                        "filename": "user_item_hyperedge_list.pkl",
                        "md5": user_item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "gowalla.train.rating",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "gowalla.test.rating",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "fri_train_list": {
                "upon": [
                    {
                        "filename": "trainFriendEdges.txt",
                        "md5": fri_train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep="\t"),
            },
            "fri_test_list": {
                "upon": [
                    {
                        "filename": "testFriendEdges.txt",
                        "md5": fri_test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep="\t"),
            }
        }

class GowallaSeq(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("gowalla",data_root)
        with open(data_root+'gowalla/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'gowalla/itemInfo.json', 'r') as f:
            item_config = json.load(f)
        with open(data_root+'gowalla/userItemInfo.json', 'r') as f:
            user_item_config = json.load(f)

        with open(data_root+'gowalla/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'gowalla/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'gowalla/user_item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/gowalla.seq.test.target', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/gowalla.seq.train.seq', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/testFriendEdges.txt', 'rb') as f:
            data = f.read()
        fri_test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/trainFriendEdges.txt', 'rb') as f:
            data = f.read()
        fri_train_list_md5 = hashlib.md5(data).hexdigest()

        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # self.train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        self.test_items = load_test_rating_as_dict(data_root+'gowalla/gowalla.seq.test.target')

        self.neg_items = load_test_negative_as_dict(data_root+'gowalla/gowalla.seq.test.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_item_vertices": user_item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "num_user_item_edges": user_item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "user_item_edge_list": {
                "upon": [
                    {
                        "filename": "user_item_hyperedge_list.pkl",
                        "md5": user_item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "gowalla.seq.train.seq",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "gowalla.seq.test.target",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "fri_train_list": {
                "upon": [
                    {
                        "filename": "trainFriendEdges.txt",
                        "md5": fri_train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep="\t"),
            },
            "fri_test_list": {
                "upon": [
                    {
                        "filename": "testFriendEdges.txt",
                        "md5": fri_test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep="\t"),
            }
        }
class GowallaICS(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("gowalla",data_root)
        with open(data_root+'gowalla/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'gowalla/itemInfo.json', 'r') as f:
            item_config = json.load(f)
        with open(data_root+'gowalla/userItemInfo.json', 'r') as f:
            user_item_config = json.load(f)

        with open(data_root+'gowalla/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'gowalla/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'gowalla/user_item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/gowalla.test.ics.rating', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/gowalla.train.ics.rating', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/testFriendEdges.txt', 'rb') as f:
            data = f.read()
        fri_test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/trainFriendEdges.txt', 'rb') as f:
            data = f.read()
        fri_train_list_md5 = hashlib.md5(data).hexdigest()

        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        # self.train_
        self.test_items = load_test_rating_as_dict(data_root+'gowalla/gowalla.test.ics.rating')

        self.neg_items = load_test_negative_as_dict(data_root+'gowalla/gowalla.test.ics.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_item_vertices": user_item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "num_user_item_edges": user_item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "user_item_edge_list": {
                "upon": [
                    {
                        "filename": "user_item_hyperedge_list.pkl",
                        "md5": user_item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "gowalla.train.ics.rating",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "gowalla.test.ics.rating",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "fri_train_list": {
                "upon": [
                    {
                        "filename": "trainFriendEdges.txt",
                        "md5": fri_train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep="\t"),
            },
            "fri_test_list": {
                "upon": [
                    {
                        "filename": "testFriendEdges.txt",
                        "md5": fri_test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep="\t"),
            }
        }
class GowallaUCS(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("gowalla",data_root)
        with open(data_root+'gowalla/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'gowalla/itemInfo.json', 'r') as f:
            item_config = json.load(f)
        with open(data_root+'gowalla/userItemInfo.json', 'r') as f:
            user_item_config = json.load(f)

        with open(data_root+'gowalla/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'gowalla/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'gowalla/user_item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/gowalla.test.ucs.rating', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/gowalla.train.ucs.rating', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/testFriendEdges.txt', 'rb') as f:
            data = f.read()
        fri_test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'gowalla/trainFriendEdges.txt', 'rb') as f:
            data = f.read()
        fri_train_list_md5 = hashlib.md5(data).hexdigest()

        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        # self.train_
        self.test_items = load_test_rating_as_dict(data_root+'gowalla/gowalla.test.ucs.rating')

        self.neg_items = load_test_negative_as_dict(data_root+'gowalla/gowalla.test.ucs.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_item_vertices": user_item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "num_user_item_edges": user_item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "user_item_edge_list": {
                "upon": [
                    {
                        "filename": "user_item_hyperedge_list.pkl",
                        "md5": user_item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "gowalla.train.ucs.rating",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "gowalla.test.ucs.rating",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "fri_train_list": {
                "upon": [
                    {
                        "filename": "trainFriendEdges.txt",
                        "md5": fri_train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep="\t"),
            },
            "fri_test_list": {
                "upon": [
                    {
                        "filename": "testFriendEdges.txt",
                        "md5": fri_test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep="\t"),
            }
        }


class Foursquare(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("foursquare",data_root)
        with open(data_root+'foursquare/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'foursquare/itemInfo.json', 'r') as f:
            item_config = json.load(f)
        with open(data_root+'foursquare/userItemInfo.json', 'r') as f:
            user_item_config = json.load(f)

        with open(data_root+'foursquare/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'foursquare/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'foursquare/user_neighbors.pkl', 'rb') as f:
            data = f.read()
        user_neighbors_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'foursquare/foursquare.test.rating', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'foursquare/foursquare.train.rating', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()


        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        # self.train_
        self.test_items = load_test_rating_as_dict(data_root+'foursquare/foursquare.test.rating')

        self.neg_items = load_test_negative_as_dict(data_root+'foursquare/foursquare.test.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_item_vertices": user_item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "num_user_item_edges": user_item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "user_neighbors": {
                "upon": [
                    {
                        "filename": "user_neighbors.pkl",
                        "md5": user_neighbors_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "foursquare.train.rating",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "foursquare.test.rating",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            }
        }

class FoursquareSeq(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("foursquare",data_root)
        with open(data_root+'foursquare/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'foursquare/itemInfo.json', 'r') as f:
            item_config = json.load(f)
        with open(data_root+'foursquare/userItemInfo.json', 'r') as f:
            user_item_config = json.load(f)

        with open(data_root+'foursquare/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'foursquare/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'foursquare/user_item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'foursquare/foursquare.seq.test.target', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'foursquare/foursquare.seq.train.seq', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()


        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # self.train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        self.test_items = load_test_rating_as_dict(data_root+'foursquare/foursquare.seq.test.target')

        self.neg_items = load_test_negative_as_dict(data_root+'foursquare/foursquare.seq.test.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_item_vertices": user_item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "num_user_item_edges": user_item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "user_item_edge_list": {
                "upon": [
                    {
                        "filename": "user_item_hyperedge_list.pkl",
                        "md5": user_item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "foursquare.seq.train.seq",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "foursquare.seq.test.target",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            }
        }

class FoursquareICS(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("foursquare",data_root)
        with open(data_root+'foursquare/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'foursquare/itemInfo.json', 'r') as f:
            item_config = json.load(f)
        with open(data_root+'foursquare/userItemInfo.json', 'r') as f:
            user_item_config = json.load(f)

        with open(data_root+'foursquare/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'foursquare/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'foursquare/user_item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'foursquare/foursquare.test.ics.rating', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'foursquare/foursquare.train.ics.rating', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()

        # with open(data_root+'foursquare/testFriendEdges.txt', 'rb') as f:
        #     data = f.read()
        # fri_test_list_md5 = hashlib.md5(data).hexdigest()
        #
        # with open(data_root+'foursquare/trainFriendEdges.txt', 'rb') as f:
        #     data = f.read()
        # fri_train_list_md5 = hashlib.md5(data).hexdigest()

        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        # self.train_
        self.test_items = load_test_rating_as_dict(data_root+'foursquare/foursquare.test.ics.rating')

        self.neg_items = load_test_negative_as_dict(data_root+'foursquare/foursquare.test.ics.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_item_vertices": user_item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "num_user_item_edges": user_item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "user_item_edge_list": {
                "upon": [
                    {
                        "filename": "user_item_hyperedge_list.pkl",
                        "md5": user_item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "foursquare.train.ics.rating",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "foursquare.test.ics.rating",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },

        }

class FoursquareUCS(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("foursquare",data_root)
        with open(data_root+'foursquare/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'foursquare/itemInfo.json', 'r') as f:
            item_config = json.load(f)
        with open(data_root+'foursquare/userItemInfo.json', 'r') as f:
            user_item_config = json.load(f)

        with open(data_root+'foursquare/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'foursquare/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'foursquare/user_item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'foursquare/foursquare.test.ucs.rating', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'foursquare/foursquare.train.ucs.rating', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()

        # with open(data_root+'foursquare/testFriendEdges.txt', 'rb') as f:
        #     data = f.read()
        # fri_test_list_md5 = hashlib.md5(data).hexdigest()

        # with open(data_root+'foursquare/trainFriendEdges.txt', 'rb') as f:
        #     data = f.read()
        # fri_train_list_md5 = hashlib.md5(data).hexdigest()

        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        # self.train_
        self.test_items = load_test_rating_as_dict(data_root+'foursquare/foursquare.test.ucs.rating')

        self.neg_items = load_test_negative_as_dict(data_root+'foursquare/foursquare.test.ucs.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_item_vertices": user_item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "num_user_item_edges": user_item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "user_item_edge_list": {
                "upon": [
                    {
                        "filename": "user_item_hyperedge_list.pkl",
                        "md5": user_item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "foursquare.train.ucs.rating",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "foursquare.test.ucs.rating",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },

        }


class AmazonICS(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("amazon",data_root)
        with open(data_root+'amazon/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'amazon/itemInfo.json', 'r') as f:
            item_config = json.load(f)

        with open(data_root+'amazon/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'amazon/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/amazon.test.ics.rating', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/amazon.train.ics.rating', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()


        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        # self.train_
        self.test_items = load_test_rating_as_dict(data_root+'amazon/amazon.test.ics.rating')

        self.neg_items = load_test_negative_as_dict(data_root+'amazon/amazon.test.ics.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "amazon.train.ics.rating",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "amazon.test.ics.rating",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            }
        }

class AmazonUCS(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("amazon",data_root)
        with open(data_root+'amazon/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'amazon/itemInfo.json', 'r') as f:
            item_config = json.load(f)

        with open(data_root+'amazon/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'amazon/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/amazon.test.ucs.rating', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/amazon.train.ucs.rating', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()



        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        # self.train_
        self.test_items = load_test_rating_as_dict(data_root+'amazon/amazon.test.ucs.rating')

        self.neg_items = load_test_negative_as_dict(data_root+'amazon/amazon.test.ucs.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],

            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },

            "train_list": {
                "upon": [
                    {
                        "filename": "amazon.train.ucs.rating",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "amazon.test.ucs.rating",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
        }



class Amazon(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("amazon",data_root)
        with open(data_root+'amazon/userInfo2.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'amazon/itemInfo2.json', 'r') as f:
            item_config = json.load(f)

        with open(data_root+'amazon/user_hyperedge_list2.pkl', 'rb') as f:
            data = f.read()
        user_hyedge2_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/item_hyperedge_list2.pkl', 'rb') as f:
            data = f.read()
        item_hyedge2_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/amazon.test.rating', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/amazon.train.rating', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()


        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        # self.train_
        self.test_items = load_test_rating_as_dict(data_root+'amazon/amazon.test.rating')

        self.neg_items = load_test_negative_as_dict(data_root+'amazon/amazon.test.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "user_edge_list2": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list2.pkl",
                        "md5": user_hyedge2_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list2": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list2.pkl",
                        "md5": item_hyedge2_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "amazon.train.rating",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "amazon.test.rating",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
        }

class AmazonSeq(BaseData):
    def __init__(self,data_root='your path data root'):
        super().__init__("amazon",data_root)
        with open(data_root+'amazon/userInfo.json', 'r') as f:
            user_config = json.load(f)
        with open(data_root+'amazon/itemInfo.json', 'r') as f:
            item_config = json.load(f)
        # with open(data_root+'amazon/userItemInfo.json', 'r') as f:
        #     user_item_config = json.load(f)

        with open(data_root+'amazon/user_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        user_hyedge_md5 = hashlib.md5(data).hexdigest()
        with open(data_root+'amazon/item_hyperedge_list.pkl', 'rb') as f:
            data = f.read()
        item_hyedge_md5 = hashlib.md5(data).hexdigest()
        # with open(data_root+'amazon/user_item_hyperedge_list.pkl', 'rb') as f:
        #     data = f.read()
        # user_item_hyedge_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/amazon.seq.test.target', 'rb') as f:
            data = f.read()
        test_list_md5 = hashlib.md5(data).hexdigest()

        with open(data_root+'amazon/amazon.seq.train.seq', 'rb') as f:
            data = f.read()
        train_list_md5 = hashlib.md5(data).hexdigest()


        # traindf = pd.read_csv(data_root+'gowalla/gowalla.train.rating',sep=' ',header=None,names=['user','item'])
        # testdf = pd.read_csv(data_root+'gowalla/gowalla.test.rating',sep=' ',header=None,names=['user','item'])
        # self.train_items = traindf.groupby('user')['item'].apply(list).to_dict()
        self.test_items = load_test_rating_as_dict(data_root+'amazon/amazon.seq.test.target')

        self.neg_items = load_test_negative_as_dict(data_root+'amazon/amazon.seq.test.negative')



        self._content = {
            "num_classes": 1,
            "num_user_vertices": user_config['num_vertices'],
            "num_item_vertices": item_config['num_vertices'],
            "num_user_edges": user_config['num_hyperedge'],
            "num_item_edges": item_config['num_hyperedge'],
            "user_edge_list": {
                "upon": [
                    {
                        "filename": "user_hyperedge_list.pkl",
                        "md5": user_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "item_edge_list": {
                "upon": [
                    {
                        "filename": "item_hyperedge_list.pkl",
                        "md5": item_hyedge_md5,
                        'bk_url': None
                    }
                ],
                "loader": load_from_pickle
            },
            "train_list": {
                "upon": [
                    {
                        "filename": "amazon.seq.train.seq",
                        "md5": train_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_list": {
                "upon": [
                    {
                        "filename": "amazon.seq.test.target",
                        "md5": test_list_md5,
                        'bk_url': None
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            }
        }
