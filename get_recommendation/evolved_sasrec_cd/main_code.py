import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import random
import numpy as np
import json
import importlib.util
from pathlib import Path
import math
from util_functions import *

from openai import OpenAI
from models import Model


def str2bool(s):
    if s not in {"false", "true", "False", "True"}:
        raise ValueError("Not a valid boolean string")
    elif s == "true" or s == "True":
        return True
    else:
        return False


def main(args):

    args.dataset = "./../../data_cache/CDs_and_Vinyl"
    dataset = data_partition(args.dataset)
    max_epoch_seconds = 30 * 60
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    if not os.path.exists(f"{args.dataset}/validation_candidate.json"):
        validation_candidate, test_candidate = make_candidate(
            user_train, user_valid, user_test, itemnum
        )
        with open(
            f"{args.dataset}/validation_candidate.json", "w", encoding="utf-8"
        ) as f:
            json.dump(validation_candidate, f, ensure_ascii=False, indent=4)
        with open(f"{args.dataset}/test_candidate.json", "w", encoding="utf-8") as f:
            json.dump(test_candidate, f, ensure_ascii=False, indent=4)
    else:
        with open(f"{args.dataset}/validation_candidate.json", "r") as j_file:
            validation_candidate = json.load(j_file)
        with open(f"{args.dataset}/test_candidate.json", "r") as j_file:
            test_candidate = json.load(j_file)

    validation_candidate_shuffle = {}
    validation_user_shuffle = {}
    for k, v in validation_candidate.items():
        u_s = [i for i in range(len(v))]
        random.shuffle(u_s)
        validation_candidate_shuffle[k] = [v[u_ss] for u_ss in u_s]
        validation_user_shuffle[k] = u_s.index(0)

    test_candidate_shuffle = {}
    test_user_shuffle = {}
    for k, v in test_candidate.items():
        u_s = [i for i in range(len(v))]
        random.shuffle(u_s)
        test_candidate_shuffle[k] = [v[u_ss] for u_ss in u_s]
        test_user_shuffle[k] = u_s.index(0)
    
    cc = 0.0
    for u, v in user_train["History"].items():
        cc += len(v)
    print("average sequence length: %.2f" % (cc / len(user_train["History"])))

    train_data_set = SeqDataset(user_train, itemnum, args.maxlen)
    valid_data_set = SeqDataset_Validation(
        user_train,
        user_valid,
        itemnum,
        args.maxlen,
        validation_candidate_shuffle,
        validation_user_shuffle,
    )
    test_data_set = SeqDataset_Test(
        user_train,
        user_valid,
        user_test,
        itemnum,
        args.maxlen,
        test_candidate_shuffle,
        test_user_shuffle
    )
    ### >>> Self_EvolveRec-BLOCK-START: DataLoader tuned for GPU throughput, enable shuffle for train
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=("cuda" in str(args.device)),
        num_workers=2,
        persistent_workers=True,
    )
    valid_data_loader = DataLoader(
        valid_data_set,
        batch_size=args.batch_size,
        pin_memory=("cuda" in str(args.device)),
        num_workers=2,
        persistent_workers=True,
    )
    test_data_loader = DataLoader(
        test_data_set,
        batch_size = args.batch_size,
        pin_memory=("cuda" in str(args.device)),
        num_workers=2,
        persistent_workers=True,
    )   

    ### <<< Self_EvolveRec-BLOCK-END

    early_stop = 0
    model = Model(usernum, itemnum, args, user_train).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    ### >>> Self_EvolveRec-BLOCK-START: Zero-out padding rows for all embeddings
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    if hasattr(model, "cat_emb"):
        with torch.no_grad():
            model.cat_emb.weight.data[0, :] = 0
    if hasattr(model, "price_emb"):
        with torch.no_grad():
            model.price_emb.weight.data[0, :] = 0
    ### <<< Self_EvolveRec-BLOCK-END

    model.train()

    epoch_start_idx = 1

    ### >>> Self_EvolveRec-BLOCK-START: Enable AMP GradScaler for mixed precision (updated API) with AdamW for better generalization
    adam_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in str(args.device)))
    ### <<< Self_EvolveRec-BLOCK-END

    best_val_acc = 0.0
    best_valid_ndcg, best_valid_hr = 0.0, 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break
        if early_stop > 3:
            break
        train_iterator = tqdm(
            train_data_loader, desc="Training (epoch X) (loss=X.X)", dynamic_ncols=True
        )
        train_epoch_start = time.time()
        train_running_batch_time = 0.0
        train_timed_batches = 0
        for step, data in enumerate(train_iterator):
            batch_start = time.time()
            u, seq, pos, neg = data
            u = list(map(int, u))
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            ### >>> Self_EvolveRec-BLOCK-START: Mixed precision training and gradient clipping (updated API + robust NaN/Inf guard)
            adam_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=("cuda" in str(args.device))):
                loss = model(u, seq, pos, neg)

            if not torch.isfinite(loss).item():
                return -1, "Nan"

            scaler.scale(loss).backward()
            scaler.unscale_(adam_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(adam_optimizer)
            scaler.update()
            ### <<< Self_EvolveRec-BLOCK-END
            train_iterator.set_description(
                f"Training (epoch {epoch}) (loss={loss.item():.4f})"
            )
            batch_end = time.time()
            batch_time = batch_end - batch_start
            train_running_batch_time += batch_time
            train_timed_batches += 1


        if epoch % 5 == 0:
            model.eval()
            t1 = time.time() - t0
            print("Validating")
            valid_user, NDCG, HT = 0, 0, 0
            valid_iterator = tqdm(
                valid_data_loader,
                desc="Validation (HR@5=X.X) (NDCG@5=X.X)",
                dynamic_ncols=True,
            )
            valid_running_batch_time = 0.0
            valid_timed_batches = 0

            for step, data in enumerate(valid_iterator):
                batch_start = time.time()
                u, seq, candi, shuf_ind = data
                u = list(map(int, u))
                u, seq, candi = np.array(u), np.array(seq), np.array(candi)
                ### >>> Self_EvolveRec-BLOCK-START: No-grad inference for validation to save memory/compute
                with torch.no_grad():
                    logits = model(u, seq, item_indices=candi)
                    target_item_indices = shuf_ind.to(args.device)
                    HT_score, NDCG_score, n_user = get_score(
                        logits, target_item_indices
                    )
                ### <<< Self_EvolveRec-BLOCK-END
                HT += HT_score
                NDCG += NDCG_score
                valid_user += n_user
                valid_iterator.set_description(
                    f"Validation (HR@5={HT/max(valid_user,1):.4f}) (NDCG@5={NDCG/max(valid_user,1):.4f})"
                )
                batch_end = time.time()
                batch_time = batch_end - batch_start
                valid_running_batch_time += batch_time
                valid_timed_batches += 1

            t_valid = [HT / max(valid_user, 1), NDCG / max(valid_user, 1)]

            if math.trunc(t_valid[0] * 10000) > math.trunc(best_val_acc * 10000):
                best_val_acc = max(t_valid[0], best_val_acc)
                best_model = copy.deepcopy(model.state_dict())
                best_valid_ndcg = t_valid[1]
                best_valid_hr = t_valid[0]
                early_stop = 0
            else:
                early_stop += 1
            t0 = time.time()
            model.train()

    print(f"Best Validation HR@5: {best_valid_hr:.4f}")
    print(f"Best Validation NDCG@5: {best_valid_ndcg:.4f}")

    ### >>> Self_EvolveRec-BLOCK-START: Load best checkpoint before evaluation if available
    if "best_model" in locals():
        try:
            model.load_state_dict(best_model)
        except Exception as e:
            print(f"Warning: could not load best_model state_dict: {e}")
    else:
        print("Warning: best_model not set; using last-epoch model for evaluation.")
    model.load_state_dict(best_model)
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END
    # Mathematical Diagnosis

    # User Centric Diagnosis
    print("Testing")
    test_user, NDCG_Test, HT_Test = 0, 0, 0
    test_iterator = tqdm(
        test_data_loader,
        desc="Test (HR@5=X.X) (NDCG@5=X.X)",
        dynamic_ncols=True,
    )
    valid_running_batch_time = 0.0
    valid_timed_batches = 0

    for step, data in enumerate(test_iterator):
        batch_start = time.time()
        u, seq, candi, shuf_ind = data
        u = list(map(int, u))
        u, seq, candi = np.array(u), np.array(seq), np.array(candi)
        logits = model(u, seq, item_indices=candi)
        target_item_indices = shuf_ind.to(args.device)
        HT_score, NDCG_score, n_user = get_score(logits, target_item_indices)
        HT_Test += HT_score
        NDCG_Test += NDCG_score
        test_user += n_user
        test_iterator.set_description(
            f"Test (HR@5={HT_Test/max(test_user,1):.4f}) (NDCG@5={NDCG_Test/max(test_user,1):.4f})"
        )
        batch_end = time.time()
        batch_time = batch_end - batch_start
        valid_running_batch_time += batch_time
        valid_timed_batches += 1

    t_test = [HT_Test / max(test_user, 1), NDCG_Test / max(test_user, 1)]
    best_test_ndcg = t_test[1]
    best_test_hr = t_test[0]
    print(f"Best Test HR@5: {best_test_hr:.4f}")
    print(f"Best Test NDCG@5: {best_test_ndcg:.4f}")

if __name__ == "__main__":
    args = get_args()
    results = main(args)


