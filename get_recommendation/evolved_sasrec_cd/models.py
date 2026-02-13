### >>> Self_EvolveRec-BLOCK-START: Add F import for custom losses and ops, warnings for safe diagnostics
import numpy as np
import torch
from tqdm import tqdm
import random
import json
import torch.nn.functional as F
import warnings

### <<< Self_EvolveRec-BLOCK-END


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
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

        with open(f"{eval_args.dataset}/meta.json", "r", encoding="utf-8") as fs:
            meta_data = json.load(fs)

        self.meta_data = meta_data

        with open(
            f"{eval_args.dataset}/train_review.json", "r", encoding="utf-8"
        ) as fs:
            review_data = json.load(fs)
        self.review_data = review_data

        ### >>> Self_EvolveRec-BLOCK-START: Multimodal fusion (category/price), popularity stats, side gating, recency pooling, inference penalties, cached causal mask, + VICReg projector and regularizer weights
        self.user_train = user_train

        # Build lightweight side-information vocabularies and per-item indices
        # Category vocab (id 0 = unknown)
        cat2id = {"UNK": 0}
        item2cat = [0] * (self.item_num + 1)

        # Price buckets (id 0 = unknown)
        def price_to_bucket(p):
            try:
                if p is None:
                    return 0
                pf = float(str(p).replace("$", "").replace(",", "").strip())
                if pf < 10:
                    return 1
                if pf < 20:
                    return 2
                if pf < 50:
                    return 3
                if pf < 100:
                    return 4
                return 5
            except Exception:
                return 0

        item2price = [0] * (self.item_num + 1)
        for i in range(1, self.item_num + 1):
            md = self.meta_data.get(str(i), {})
            cats = md.get("categories", [])
            cat_name = cats[0] if isinstance(cats, list) and len(cats) > 0 else "UNK"
            if cat_name not in cat2id:
                cat2id[cat_name] = len(cat2id)
            item2cat[i] = cat2id.get(cat_name, 0)
            item2price[i] = price_to_bucket(md.get("price", None))

        self.num_cats = len(cat2id)
        self.num_price_bins = 6  # 0..5 as defined above

        # Embeddings
        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, eval_args.hidden_units, padding_idx=0
        )
        self.cat_emb = torch.nn.Embedding(
            self.num_cats, eval_args.hidden_units, padding_idx=0
        )
        self.price_emb = torch.nn.Embedding(
            self.num_price_bins, eval_args.hidden_units, padding_idx=0
        )

        # Buffers for fast lookup (move with .to(device))
        self.register_buffer("item2cat", torch.tensor(item2cat, dtype=torch.long))
        self.register_buffer("item2price", torch.tensor(item2price, dtype=torch.long))

        # Popularity statistics for weighting and inference debiasing
        pop_counts = torch.zeros(self.item_num + 1, dtype=torch.float)
        hist_dict = (
            self.user_train.get("History", {})
            if isinstance(self.user_train, dict)
            else {}
        )
        if isinstance(hist_dict, dict):
            for u, hist in hist_dict.items():
                if isinstance(hist, (list, tuple)) and len(hist) > 0:
                    ids = torch.as_tensor(hist, dtype=torch.long)
                    ids = ids[(ids > 0) & (ids <= self.item_num)]
                    if ids.numel() > 0:
                        pop_counts.index_add_(
                            0, ids, torch.ones_like(ids, dtype=torch.float)
                        )
        else:
            warnings.warn(
                "user_train['History'] is not a dict; popularity stats set to zeros."
            )
        pop_counts[0] = 0.0
        # Avoid zeros for weights; alpha=0.5 by default
        alpha = 0.5
        pop_for_weight = torch.clamp(pop_counts, min=1.0)
        pos_weight = 1.0 / (pop_for_weight.pow(alpha))
        pos_weight = pos_weight / (pos_weight.mean().clamp_min(1e-6))
        self.register_buffer("pos_weight", pos_weight)

        # Normalized popularity for inference penalty [0,1]
        if pop_counts.max() > pop_counts.min():
            pop_norm = (pop_counts - pop_counts.min()) / (
                pop_counts.max() - pop_counts.min()
            )
        else:
            pop_norm = torch.zeros_like(pop_counts)
        self.register_buffer("pop_norm", pop_norm)
        # Inference-time adjustments: popularity debiasing, category/price/duplicate penalties
        self.infer_pop_beta = 0.05
        self.infer_cat_beta = 0.05
        self.infer_dup_beta = 0.02
        self.infer_price_beta = 0.02

        # Lightweight projector/gating for side features
        self.side_ln = torch.nn.LayerNorm(eval_args.hidden_units)
        self.side_alpha = torch.nn.Parameter(torch.tensor(0.5))

        self.pos_emb = torch.nn.Embedding(
            eval_args.maxlen + 1, eval_args.hidden_units, padding_idx=0
        )
        self.emb_dropout = torch.nn.Dropout(p=eval_args.dropout_rate)

        # Recency-aware pooling over last-k items with learnable decay
        self.pool_k = min(eval_args.maxlen, 5)
        self.recency_log_tau = torch.nn.Parameter(
            torch.tensor(0.0)
        )  # tau=softplus(log_tau)

        # Cached causal attention mask for speed (maxlen x maxlen, True means masked)
        causal = ~torch.tril(
            torch.ones((eval_args.maxlen, eval_args.maxlen), dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal)

        # Projector head for VICReg on pooled representations
        H = eval_args.hidden_units
        proj_out = max(32, H // 2)
        self.proj_out_dim = proj_out
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(H, H),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(H, proj_out),
        )

        # Regularizer weights (small, stable defaults)
        self.alpha_vic = 0.02
        self.lam_consistency = 0.05
        self.lambda_attn_entropy = 0.001
        ### <<< Self_EvolveRec-BLOCK-END

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(eval_args.hidden_units, eps=1e-8)

        Number_of_Layer = 2
        Number_of_Head = 2
        for _ in range(
            Number_of_Layer
        ):  # You can adjust the number of blocks likes, 3, 4, ...
            new_attn_layernorm = torch.nn.LayerNorm(eval_args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                eval_args.hidden_units,
                Number_of_Head,  # This is the number of heads of attention layer. You can adjust the number of heads as you want.
                eval_args.dropout_rate,
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(eval_args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                eval_args.hidden_units, eval_args.dropout_rate
            )
            self.forward_layers.append(new_fwd_layer)

    ### >>> Self_EvolveRec-BLOCK-START: Device-optimized seq emb + side-feature fusion + recency pooling (use cached causal mask)
    def log2feats(self, log_seqs):
        # Build tensors directly on device to avoid CPU->GPU ping-pong
        log_seqs_t = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)
        seqs = self.get_item_rep(log_seqs_t)
        seqs *= self.item_emb.embedding_dim**0.5

        B, L = log_seqs_t.shape
        pos_ids = torch.arange(1, L + 1, device=self.dev).unsqueeze(0).expand(B, -1)
        mask_nonzero = log_seqs_t.ne(0).long()
        pos_ids = pos_ids * mask_nonzero
        seqs += self.pos_emb(pos_ids)
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = self.causal_mask[:tl, :tl]

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            mha_outputs, _ = self.attention_layers[i](
                seqs, seqs, seqs, attn_mask=attention_mask
            )
            seqs = self.attention_layernorms[i](seqs + mha_outputs)
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def pool_user(self, log_feats, nonpad_mask):
        """
        Recency-aware pooling over last-k valid positions with an exponential decay.
        log_feats: (B, L, H)
        nonpad_mask: (B, L) bool where True marks valid tokens
        returns: (B, H)
        """
        B, L, H = log_feats.shape
        idx = torch.arange(1, L + 1, device=self.dev).unsqueeze(0).expand(B, -1)  # 1..L
        dist_from_end = L - idx + 1  # 1 for last, 2 for second last, ...
        # Exponential decay with learnable tau > 0
        tau = F.softplus(self.recency_log_tau) + 1e-4
        w = torch.exp(-tau * dist_from_end)
        # keep only last-k indices if k > 0
        if self.pool_k > 0:
            lastk_mask = dist_from_end <= self.pool_k
            w = w * lastk_mask.float()
        # mask out pads and normalize
        w = w * nonpad_mask.float()
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        w = w / denom
        pooled = (w.unsqueeze(-1) * log_feats).sum(dim=1)
        return pooled

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Fused item representation and scoring
    def get_item_rep(self, item_indices):
        """
        Returns fused item representation: id_emb + side_emb (category/price gated).
        Works with arbitrary-shaped index tensors.
        """
        id_embs = self.item_emb(item_indices)
        # Side features: category + price
        cat_ids = self.item2cat[item_indices]
        price_ids = self.item2price[item_indices]
        side = self.cat_emb(cat_ids) + self.price_emb(price_ids)
        side = self.side_ln(side)
        fused = id_embs + self.side_alpha * side
        return fused

    def predict_score(self, user_seq_emb, item_indices):
        item_embs = self.get_item_rep(item_indices)
        user_seq_expanded = user_seq_emb.unsqueeze(2)
        logits = (user_seq_expanded * item_embs).sum(dim=-1)
        return logits

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Popularity-weighted BCE + BPR loss; inference debiasing with recency pooling and coherence penalties
    def forward(
        self,
        user_ids,
        log_seqs,
        pos_seqs=None,
        neg_seqs=None,
        item_indices=None,
    ):

        log_feats = self.log2feats(log_seqs)

        # Inference path (ranking candidates) with light coherence adjustments
        if item_indices is not None:
            seq_t = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)
            nonpad_mask = seq_t.ne(0)
            # Recency-aware pooled user representation
            pooled = self.pool_user(log_feats, nonpad_mask)
            final_feat = pooled.unsqueeze(1)
            candi = torch.as_tensor(
                item_indices, dtype=torch.long, device=self.dev
            ).unsqueeze(1)
            logits = self.predict_score(final_feat, candi).squeeze(1)
            candi2d = candi.squeeze(1)

            # Popularity debiasing
            pop_penalty = self.pop_norm[candi2d].float()
            logits = logits - self.infer_pop_beta * pop_penalty

            # Category-coherence bonus vs most recent item
            B, L = seq_t.shape
            idxs = torch.arange(L, device=self.dev).unsqueeze(0).expand(B, -1)
            masked_idx = idxs.masked_fill(~nonpad_mask, -1)
            last_idx = masked_idx.max(dim=1).values
            last_item_ids = torch.where(
                last_idx >= 0,
                seq_t[torch.arange(B, device=self.dev), last_idx.clamp_min(0)],
                torch.zeros(B, dtype=torch.long, device=self.dev),
            )
            last_cat = self.item2cat[last_item_ids]  # [B]
            candi_cats = self.item2cat[candi2d]  # [B,C]
            cat_match = (candi_cats == last_cat.unsqueeze(1)).float()
            logits = logits + self.infer_cat_beta * cat_match

            # Price mismatch penalty
            last_price = self.item2price[last_item_ids].float()  # [B]
            candi_price = self.item2price[candi2d].float()  # [B,C]
            price_diff = (candi_price - last_price.unsqueeze(1)).abs()  # [B,C]
            logits = logits - self.infer_price_beta * (price_diff / 5.0)

            # Duplicate-history penalty (avoid recommending items already seen)
            if candi2d.size(1) <= 2048:
                dup_mask = (
                    (
                        (candi2d.unsqueeze(2) == seq_t.unsqueeze(1))
                        & nonpad_mask.unsqueeze(1)
                    )
                    .any(dim=2)
                    .float()
                )
                logits = logits - self.infer_dup_beta * dup_mask

            return logits

        # Training path
        if pos_seqs is None or neg_seqs is None:
            raise ValueError(
                "Training mode requires both pos_seqs and neg_seqs when item_indices is None."
            )

        pos_t = torch.as_tensor(pos_seqs, dtype=torch.long, device=self.dev)
        neg_t = torch.as_tensor(neg_seqs, dtype=torch.long, device=self.dev)
        mask = pos_t.ne(0)

        all_items = torch.stack([pos_t, neg_t], dim=-1)
        all_logits = self.predict_score(log_feats, all_items)
        pos_logits = all_logits[:, :, 0]
        neg_logits = all_logits[:, :, 1]

        # Popularity-weighted BCE for positives (SNIPS-style renormalized per batch positions)
        pos_labels = torch.ones_like(pos_logits, dtype=torch.float, device=self.dev)
        neg_labels = torch.zeros_like(neg_logits, dtype=torch.float, device=self.dev)

        # Gather per-position positive weights and renormalize on mask
        w_pos = self.pos_weight[pos_t].clamp_min(1e-2).clamp_max(50.0)
        if mask.any():
            mean_w = w_pos[mask].mean().clamp_min(1e-6)
            w_pos = w_pos / mean_w

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits[mask], pos_labels[mask], reduction="none"
        )
        pos_loss = (
            (pos_loss * w_pos[mask]).mean()
            if mask.any()
            else torch.tensor(0.0, device=self.dev)
        )

        neg_loss = (
            F.binary_cross_entropy_with_logits(
                neg_logits[mask], neg_labels[mask], reduction="mean"
            )
            if mask.any()
            else torch.tensor(0.0, device=self.dev)
        )

        # Pairwise BPR/softplus margin to tighten separation
        bpr_lambda = 0.2
        pair_loss = (
            F.softplus(-(pos_logits[mask] - neg_logits[mask])).mean()
            if mask.any()
            else torch.tensor(0.0, device=self.dev)
        )

        loss = pos_loss + neg_loss + bpr_lambda * pair_loss
        return loss


### <<< Self_EvolveRec-BLOCK-END


