import numpy as np
import torch
from tqdm import tqdm
import random
import json

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) 
        return outputs


class Model(torch.nn.Module):
    def __init__(self, user_num, item_num, eval_args, user_train):
        super(Model, self).__init__()

        self.eval_args = eval_args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = eval_args.device
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        with open(f'{eval_args.dataset}/meta.json', 'r', encoding='utf-8') as fs:
            meta_data = json.load(fs)
            
        self.meta_data = meta_data
        
        with open(f'{eval_args.dataset}/train_review.json', 'r', encoding='utf-8') as fs:
          review_data = json.load(fs)
        self.review_data = review_data
        
        self.user_train = user_train
        
        self.item_emb = torch.nn.Embedding(self.item_num+1, eval_args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(eval_args.maxlen+1, eval_args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=eval_args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(eval_args.hidden_units, eps=1e-8)

        Number_of_Layer = 2
        Number_of_Head = 2
        for _ in range(Number_of_Layer):# You can adjust the number of blocks likes, 3, 4, ...
            new_attn_layernorm = torch.nn.LayerNorm(eval_args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(eval_args.hidden_units,
                                                            Number_of_Head,# This is the number of heads of attention layer. You can adjust the number of heads as you want.
                                                            eval_args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(eval_args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(eval_args.hidden_units, eval_args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] 
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                            attn_mask=attention_mask)
            seqs = self.attention_layernorms[i](seqs + mha_outputs)
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def predict_score(self, user_seq_emb, item_indices):
      item_embs = self.item_emb(item_indices)
      
      user_seq_expanded = user_seq_emb.unsqueeze(2)
      logits = (user_seq_expanded * item_embs).sum(dim=-1)
            
      return logits
    
    def forward(self, user_ids, log_seqs, pos_seqs=None, neg_seqs=None, item_indices=None,):
        """
        - input formats:
        - user_ids: A batch of user IDs, e.g., array([8, 10, 14, 20, 21 ...])
        - log_seqs: A batch of user interaction sequences, e.g., array([[0, 0, 25, 26, 27], [0, 55, 70, 71, 89], ...])
        - pos_seqs: A batch of user's positive item (i.e., label), e.g., array([[0, 0, 26, 27, 28], [0, 70, 71, 89, 73], ...])
        - neg_seqs: A batch of user's negative item, e.g., array([[0, 0, 18456, 133803, 75077], [0, 11520, 222222, 4, 19], ...])
        """   
        log_feats = self.log2feats(log_seqs)

        if item_indices is not None:
          final_feat = log_feats[:, -1, :].unsqueeze(1)  
          
          item_indices = torch.LongTensor(item_indices).to(self.dev).unsqueeze(1)
          logits = self.predict_score(final_feat, item_indices)
          return logits.squeeze(1)
        
        if pos_seqs is None or neg_seqs is None:
          raise ValueError("Training mode requires both pos_seqs and neg_seqs when item_indices is None.")

        indices = np.where(pos_seqs != 0)
        pos_seqs = torch.LongTensor(pos_seqs).to(self.dev)
        neg_seqs = torch.LongTensor(neg_seqs).to(self.dev)
        all_items = torch.stack([pos_seqs, neg_seqs], dim=-1)
        
        all_logits = self.predict_score(log_feats, all_items)
        
        pos_logits = all_logits[:, :, 0]
        neg_logits = all_logits[:, :, 1]
        
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        
        loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])
        
        return loss