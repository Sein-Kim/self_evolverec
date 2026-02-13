import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import json
from torch.utils.data import Dataset
from tqdm import tqdm
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

class Args:
    def __init__(self):
        # device
        self.device = 'cuda'
        self.dataset = 'CDs_and_Vinyl'
        self.train_dir = 'Recommendation_Models'
        self.batch_size = 256
        self.lr = 0.001
        self.maxlen = 200
        self.hidden_units = 50
        self.num_epochs = 300
        self.dropout_rate = 0.2
        self.l2_emb = 0.005
        self.inference_only = False
        self.state_dict_path = None
        self.norm_first = False
        self.conduct_test = False
        
def get_args():
    return Args()


class SeqDataset(Dataset):
    def __init__(self, user_train, num_item, max_len):
        self.user_train = user_train
        self.num_item = num_item
        self.max_len = max_len

        self.user_ids = list(self.user_train['History'].keys())
        print(f"Train - Initializing with {len(self.user_ids)} users.")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]

        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)

        nxt = self.user_train['History'][user_id][-1]
        length_idx = self.max_len - 1

        ts = set(self.user_train['History'][user_id])
        for i in reversed(self.user_train['History'][user_id][:-1]):
            seq[length_idx] = i
            pos[length_idx] = nxt
            if nxt != 0: neg[length_idx] = random_neq(1, self.num_item + 1, ts)
            nxt = i
            length_idx -= 1
            if length_idx == -1: break

        return user_id, seq, pos, neg


class SeqDataset_Validation(Dataset):
    def __init__(self, user_train, user_valid, num_item, max_len, candi_dict, shuffle_ind):
        self.user_train = user_train
        self.user_valid = user_valid
        self.num_item = num_item
        self.max_len = max_len
        self.candi_dict = candi_dict
        self.shuffle_ind = shuffle_ind

        self.user_ids = list(set.intersection(set(user_valid['History'].keys()),set(user_train['History'].keys())))
        print(f"Validation - Initializing with {len(self.user_ids)} users.")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]

        seq = np.zeros([self.max_len], dtype=np.int32)

        ts = self.user_train['History'][user_id]
        idx = self.max_len-1
        for t in reversed(ts):
            seq[idx] = t
            idx -=1
        rated = set(ts)
        rated.add(0)
        
        negs_list = self.candi_dict[user_id]
        

        return user_id, seq, np.array(negs_list), self.shuffle_ind[user_id]


class SeqDataset_Test(Dataset):
    def __init__(self, user_train, user_valid, user_test, num_item, max_len, candi_dict, shuffle_ind):
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.num_item = num_item
        self.max_len = max_len
        self.candi_dict = candi_dict
        self.shuffle_ind = shuffle_ind

        self.user_ids = list(set.intersection(set(user_valid['History'].keys()),set(user_train['History'].keys()),set(user_test['History'].keys())))
        print(f"Test - Initializing with {len(self.user_ids)} users.")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]

        seq = np.zeros([self.max_len], dtype=np.int32)

        ts = self.user_train['History'][user_id] + self.user_valid['History'][user_id]
        idx = self.max_len-1
        for t in reversed(ts):
            seq[idx] = t
            idx -=1
        rated = set(ts)
        rated.add(0)
        
        negs_list = self.candi_dict[user_id]
        

        return user_id, seq, np.array(negs_list), self.shuffle_ind[user_id]


def make_candidate(user_train, user_valid, user_test, num_item):
    all_items = set([i for i in range(1,num_item+1)])
    user_ids = list(set.intersection(set(user_valid['History'].keys()),set(user_train['History'].keys())))
    candi_dict = {}
    for ui in tqdm(user_ids):
        ts = user_train['History'][ui]
        rated = set(ts)
        rated.add(0)
        nxt = user_valid['History'][ui][-1]
        negs_list = random.sample(list(all_items - rated), 99)
        negs_list = [nxt] + negs_list
        candi_dict[ui] = negs_list
    
    user_ids = list(set.intersection(set(user_valid['History'].keys()),set(user_train['History'].keys()),set(user_test['History'].keys())))
    candi_dict_test = {}
    for ui in tqdm(user_ids):
        ts = user_train['History'][ui] + user_valid['History'][ui]
        rated = set(ts)
        rated.add(0)
        rated.add(nxt)
        nxt = user_test['History'][ui][-1]
        negs_list = list(all_items - rated)
        negs_list = [nxt] + negs_list
        candi_dict_test[ui] = negs_list
        
    return candi_dict, candi_dict_test
        

# train/val/test data generation
def data_partition(fname):
    # assume user/item index starting from 1
    with open(f'{fname}/train.json', 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(f'{fname}/valid.json', 'r', encoding='utf-8') as f:
        valid = json.load(f)
    with open(f'{fname}/test.json', 'r', encoding='utf-8') as f:
        test = json.load(f)
    with open(f'{fname}/user2id.json', 'r', encoding='utf-8') as f:
        user2id = json.load(f)
    with open(f'{fname}/item2id.json', 'r', encoding='utf-8') as f:
        item2id = json.load(f)
        
    usernum = max(user2id.values())
    itemnum = max(item2id.values())
    
    return [train, valid, test, usernum, itemnum]

def test_collate_fn(batch):
    """
    batch: List of (user_id, seq, negs_list)
    """

    user_ids = []
    seqs = []
    negs_lists = []
    shf_list = []

    for user_id, seq, negs, shuf_ind in batch:
        user_ids.append(user_id)
        seqs.append(torch.from_numpy(seq).long())
        negs_lists.append(torch.from_numpy(negs).long())  # 길이 다름 OK
        shf_list.append(torch.tensor([shuf_ind]).long())

    return user_ids,torch.stack(seqs, dim=0),negs_lists, shf_list

def get_score(logits, target_item_indices):
    topk_values, topk_indices = torch.topk(logits, k=logits.shape[1], dim=1, largest=True, sorted=True)
    target_item_ranks = (topk_indices == target_item_indices.unsqueeze(1)).nonzero(as_tuple=True)[1]
    user_mask = target_item_ranks < 5
    
    HT = user_mask.sum().item()
    ranks = target_item_ranks[user_mask]
    NDCG = (1 / torch.log2(ranks + 2).float()).sum().item()
    
    n_user = target_item_ranks.size(0)
    
    return HT, NDCG, n_user