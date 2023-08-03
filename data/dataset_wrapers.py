import random
from typing import List, Tuple, Optional

import torch
from dhg import Hypergraph
from dhg.utils import edge_list_to_adj_dict, adj_list_to_edge_list
from torch.utils.data import Dataset

class HyperDataset(Dataset):
    def __init__(
            self,
            num_nodes: int,
            num_nodes_item: int,

            hg: Hypergraph
    ):
        self.num_nodes = num_nodes
        self.num_nodes_item = num_nodes_item
        self.num_hyedge = len(hg.e[0])
        self.k = 50
        self.hy_dict={}

        for i in range(self.num_hyedge):
            self.hy_dict[i] = hg.N_v(i)



    def sample_triplet(self):
        r"""Sample a triple of user, positive item, and negtive item from all interactions.
        """
        hedge = random.randrange(self.num_hyedge)
        assert len(self.hy_dict[hedge]) > 0
        user = random.choice(self.hy_dict[hedge])
        pos_item = random.choice(self.hy_dict[hedge])
        if user < self.num_nodes:
            neg_item = self.sample_neg_user(hedge)
        else:
            neg_item = self.sample_neg_item(hedge)
        hedge_items = random.choices(self.hy_dict[hedge],k=self.k)
        return hedge, user, pos_item, neg_item, hedge_items

    def sample_neg_item(self, item:int):
        r"""Sample a negative item for the sepcified user.

        Args:
            ``user`` (``int``): The index of the specified user.
        """
        neg_item = random.randrange(self.num_nodes,self.num_nodes+self.num_nodes_item)
        while neg_item in self.hy_dict[item]:
            neg_item = random.randrange(self.num_nodes,self.num_nodes+self.num_nodes_item)
        return neg_item

    def sample_neg_user(self, user: int):
        r"""Sample a negative item for the sepcified user.

        Args:
            ``user`` (``int``): The index of the specified user.
        """
        neg_item = random.randrange(self.num_nodes)
        while neg_item in self.hy_dict[user]:
            neg_item = random.randrange(self.num_nodes)
        return neg_item

    def __getitem__(self, index):
        hedge, anchor, pos_item, neg_item, hedge_items = self.sample_triplet()
        return hedge, anchor, pos_item, neg_item,hedge_items

    def __len__(self):
        return self.num_hyedge


class UserItemDataset(Dataset):
    r"""The dataset class of user-item bipartite graph for recommendation task.

    Args:
        ``num_users`` (``int``): The number of users.
        ``num_items`` (``int``): The number of items.
        ``user_item_list`` (``List[Tuple[int, int]]``): The list of user-item pairs.
        ``train_user_item_list`` (``List[Tuple[int, int]]``, optional): The list of user-item pairs for training. This is only needed for testing to mask those seen items in training. Defaults to ``None``.
        ``strict_link`` (``bool``): Whether to iterate through all interactions in the dataset. If set to ``False``, in training phase the dataset will keep randomly sampling interactions until meeting the same number of original interactions. Defaults to ``True``.
        ``phase`` (``str``): The phase of the dataset can be either ``"train"`` or ``"test"``. Defaults to ``"train"``.
    """

    def __init__(
            self,
            num_users: int,
            num_items: int,
            user_item_list: List[Tuple[int, int]],
            train_user_item_list: Optional[List[Tuple[int, int]]] = None,
            strict_link: bool = True,
            phase: str = "train",
    ):

        assert phase in ["train", "test"]
        self.phase = phase
        self.num_users, self.num_items = num_users, num_items
        self.user_item_list = user_item_list
        self.adj_dict = edge_list_to_adj_dict(user_item_list)
        self.strict_link = strict_link
        if phase != "train":
            assert (
                    train_user_item_list is not None
            ), "train_user_item_list is needed for testing."
            self.train_adj_dict = edge_list_to_adj_dict(train_user_item_list)


    def sample_triplet(self):
        r"""Sample a triple of user, positive item, and negtive item from all interactions.
        """
        user = random.randrange(self.num_users)
        assert len(self.adj_dict[user]) > 0
        pos_item = random.choice(self.adj_dict[user])
        neg_item = self.sample_neg_item(user)
        return user, pos_item, neg_item

    def sample_neg_item(self, user: int):
        r"""Sample a negative item for the sepcified user.

        Args:
            ``user`` (``int``): The index of the specified user.
        """
        neg_item = random.randrange(self.num_users,self.num_users+self.num_items)
        while neg_item in self.adj_dict[user]:
            neg_item = random.randrange(self.num_users,self.num_users+self.num_items)
        return neg_item

    def __getitem__(self, index):
        r"""Return the item at the index. If the phase is ``"train"``, return the (``User``-``PositiveItem``-``NegativeItem``) triplet. If the phase is ``"test"``, return all true positive items for each user.

        Args:
            ``index`` (``int``): The index of the item.
        """
        if self.phase == "train":
            if self.strict_link:
                user, pos_item = self.user_item_list[index]
                neg_item = self.sample_neg_item(user)
            else:
                user, pos_item, neg_item = self.sample_triplet()
            return user, pos_item, neg_item
        else:
            train_mask, true_rating = (
                torch.zeros(self.num_items),
                torch.zeros(self.num_items),
            )
            train_items, true_items = self.train_adj_dict[index], self.adj_dict[index]

            train_items = [x-self.num_users for x in train_items]
            true_items = [x-self.num_users for x in true_items]
            train_mask[train_items] = float("-inf")
            true_rating[true_items] = 1.0
            return index, train_mask, true_rating

    def __len__(self):
        r"""Return the length of the dataset. If the phase is ``"train"``, return the number of interactions. If the phase is ``"test"``, return the number of users.
        """
        if self.phase == "train":
            return len(self.user_item_list)
        else:
            return self.num_users


class SeqDataset(Dataset):
    def __init__(
            self,
            user_item_list: List[Tuple[int, int]],
            strict_link: bool = True,
            phase: str = "train",
    ):

        assert phase in ["train", "test"]
        self.phase = phase
        self.user_item_list = user_item_list
        self.adj_dict = edge_list_to_adj_dict(user_item_list)
        self.strict_link = strict_link




    def __getitem__(self, index):
        r"""Return the item at the index. If the phase is ``"train"``, return the (``User``-``PositiveItem``-``NegativeItem``) triplet. If the phase is ``"test"``, return all true positive items for each user.

        Args:
            ``index`` (``int``): The index of the item.
        """
        return index,self.adj_dict[index][-1]



    def __len__(self):
        r"""Return the length of the dataset. If the phase is ``"train"``, return the number of interactions. If the phase is ``"test"``, return the number of users.
        """
        return len(self.adj_dict)

class HyperSeqDataset(Dataset):
    def __init__(
            self,
            num_users: int,
            num_items: int,
            max_len: int,
            user_item_list: List[Tuple[int, int]],
            strict_link: bool = True,
            phase: str = "train",
    ):

        assert phase in ["train", "test"]
        self.phase = phase
        self.user_item_list = user_item_list
        # self.adj_dict = edge_list_to_adj_dict(user_item_list)
        self.num_users = num_users
        self.num_items = num_items

        user_seq = dict()
        user_target = dict()

        for i in range(len(self.user_item_list)):
            if self.user_item_list[i][0] not in user_seq:
                user_seq[self.user_item_list[i][0]] = []
                user_target[self.user_item_list[i][0]] = []
            cur_user_seqs = self.user_item_list[i][1:]
            for j in range(len(cur_user_seqs)//max_len):
                user_seq[self.user_item_list[i][0]].append(cur_user_seqs[j*max_len:(j+1)*max_len-1])
                user_target[self.user_item_list[i][0]].append(cur_user_seqs[(j+1)*max_len-1])
            if len(cur_user_seqs)%max_len != 0:
                if len(cur_user_seqs[-max_len:-1]) < max_len-1:
                    user_seq[self.user_item_list[i][0]].append(cur_user_seqs[-max_len:-1]+[self.user_item_list[i][0]]*(max_len-1-len(cur_user_seqs[-max_len:-1])))
                    user_target[self.user_item_list[i][0]].append(cur_user_seqs[-1])
                else:
                    user_seq[self.user_item_list[i][0]].append(cur_user_seqs[-max_len:-1])
                    user_target[self.user_item_list[i][0]].append(cur_user_seqs[-1])
        self.user_seq = user_seq
        self.user_target = user_target

        self.strict_link = strict_link


    def sample(self):
        index = random.choice(range(len(self.user_seq)))
        idx = random.choice(range(len(self.user_seq[index])))

        neg_item = random.randrange(self.num_users,self.num_users+self.num_items)
        while neg_item == self.user_target[index][idx]:
            neg_item = random.randrange(self.num_users,self.num_users+self.num_items)
        return index,self.user_seq[index][idx],self.user_target[index][idx],neg_item

    def __getitem__(self, index):
        r"""Return the item at the index. If the phase is ``"train"``, return the (``User``-``PositiveItem``-``NegativeItem``) triplet. If the phase is ``"test"``, return all true positive items for each user.

        Args:
            ``index`` (``int``): The index of the item.
        """
        user,user_seq,user_target,neg_item = self.sample()
        return user,user_seq,user_target,neg_item



    def __len__(self):
        r"""Return the length of the dataset. If the phase is ``"train"``, return the number of interactions. If the phase is ``"test"``, return the number of users.
        """
        return len(adj_list_to_edge_list(self.user_item_list))




