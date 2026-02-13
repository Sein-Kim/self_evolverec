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
from simulator.prompts_list import *
from simulator.util_functions import *
from concurrent.futures import ThreadPoolExecutor, as_completed

from diagnosis_tools import *
from simulator.simulator import *

from openai import OpenAI
from models import Model

def str2bool(s):
    if s not in {'false', 'true', 'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    elif s == 'true' or s =='True':
        return True
    else:
        return False
    

def main(args):


    args.dataset = "data_cache/CDs_and_Vinyl"
    dataset = data_partition(args.dataset)
    max_epoch_seconds = 30 * 60
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    if not os.path.exists(f'{args.dataset}/validation_candidate.json'):
        validation_candidate, test_candidate = make_candidate(user_train, user_valid, user_test, itemnum)
        with open(f"{args.dataset}/validation_candidate.json", "w", encoding="utf-8") as f:
            json.dump(validation_candidate, f, ensure_ascii=False, indent=4)
        with open(f"{args.dataset}/test_candidate.json", "w", encoding="utf-8") as f:
            json.dump(test_candidate, f, ensure_ascii=False, indent=4)
    else:
        with open(f'{args.dataset}/validation_candidate.json', 'r') as j_file:
            validation_candidate = json.load(j_file)
        with open(f'{args.dataset}/test_candidate.json', 'r') as j_file:
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
    print('average sequence length: %.2f' % (cc / len(user_train["History"])))
    
    train_data_set = SeqDataset(user_train, itemnum, args.maxlen)
    valid_data_set = SeqDataset_Validation(user_train, user_valid, itemnum, args.maxlen, validation_candidate_shuffle, validation_user_shuffle)
    test_data_set = SeqDataset_Test(user_train, user_valid, user_test, itemnum, args.maxlen, test_candidate_shuffle, test_user_shuffle)
    train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size, pin_memory=True)       
    valid_data_loader = DataLoader(valid_data_set, batch_size = args.batch_size, pin_memory=True)       
    test_data_loader = DataLoader(test_data_set, batch_size = args.batch_size, pin_memory=True)    
    
    early_stop = 0
    model = Model(usernum, itemnum, args, user_train).to(args.device)
    simulator = Simulator('gpt-5-mini', model)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    
    model.train()
    
    epoch_start_idx = 1
    
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_acc = 0.0
    best_valid_ndcg, best_valid_hr = 0.0, 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break
        if early_stop >3:
            break
        train_iterator = tqdm(train_data_loader, desc="Training (epoch X) (loss=X.X)", dynamic_ncols=True)
        train_epoch_start = time.time()
        train_running_batch_time = 0.0
        train_timed_batches = 0
        for step, data in enumerate(train_iterator):
            batch_start = time.time()
            u, seq, pos, neg = data
            u = list(map(int, u))
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            loss = model(u, seq, pos, neg)

            if torch.isnan(loss):
                return -1, "Nan"
            adam_optimizer.zero_grad()

            loss.backward()
            adam_optimizer.step()
            train_iterator.set_description(f"Training (epoch {epoch}) (loss={loss.item():.4f})")
            batch_end = time.time()
            batch_time = batch_end - batch_start
            train_running_batch_time += batch_time
            train_timed_batches += 1
            
            if step >5:
                avg_batch_time = train_running_batch_time/train_timed_batches
                if avg_batch_time *len(train_data_loader) > max_epoch_seconds:
                    time_min = avg_batch_time *len(train_data_loader)/60.0
                    return 1, {
                        "ndcg_score(0-1)": 0,
                        "hr_score(0-1)": 0,
                        "combined_score":(0.6*0 + 0.4*0),
                        "comment": train_time_prompt.format(time=time_min)
                        }
        
        if epoch % 5 ==0:
            model.eval()
            t1 = time.time() - t0
            print('Validating')
            valid_user, NDCG, HT = 0, 0, 0
            valid_iterator = tqdm(valid_data_loader, desc="Validation (HR@5=X.X) (NDCG@5=X.X)", dynamic_ncols=True)
            valid_running_batch_time = 0.0
            valid_timed_batches = 0

            for step, data in enumerate(valid_iterator):
                batch_start = time.time() 
                u, seq, candi, shuf_ind = data
                u = list(map(int, u))
                u, seq, candi = np.array(u), np.array(seq), np.array(candi)
                logits = model(u, seq, item_indices=candi)
                target_item_indices = shuf_ind.to(args.device)
                HT_score, NDCG_score, n_user = get_score(logits, target_item_indices)
                HT+=HT_score
                NDCG+=NDCG_score
                valid_user+=n_user
                valid_iterator.set_description(f"Validation (HR@5={HT/max(valid_user,1):.4f}) (NDCG@5={NDCG/max(valid_user,1):.4f})")
                batch_end = time.time()
                batch_time = batch_end - batch_start
                valid_running_batch_time += batch_time
                valid_timed_batches += 1
                if step >5:
                    avg_batch_time = valid_running_batch_time/valid_timed_batches
                    if avg_batch_time *len(valid_data_loader) > max_epoch_seconds:
                        time_min = avg_batch_time *len(valid_data_loader)/60.0
                        return 1, {
                            "ndcg_score(0-1)": 0,
                            "hr_score(0-1)": 0,
                            "combined_score":(0.6*0 + 0.4*0),
                            "comment": validation_time_prompt.format(time=time_min)
                            }
                
            t_valid = [HT/max(valid_user,1), NDCG/max(valid_user,1)]
                
            if math.trunc(t_valid[0]*10000) > math.trunc(best_val_acc*10000):
                best_val_acc = max(t_valid[0], best_val_acc)
                best_model = copy.deepcopy(model.state_dict())
                best_valid_ndcg = t_valid[1]
                best_valid_hr = t_valid[0]
                early_stop = 0
            else:
                early_stop +=1
            t0 = time.time()
            model.train()

    print(f"Best Validation HR@5: {best_valid_hr:.4f}")
    print(f"Best Validation NDCG@5: {best_valid_ndcg:.4f}")
    
    model.load_state_dict(best_model)
    model.eval()
    probe = DiagnosisProbe(model, args.device)
    # Mathematical Diagnosis
    math_diagnosis = probe.run_full_diagnosis(train_data_loader)
    math_results = math_diagnosis_agent(diagnosis_interpreter_prompt(math_diagnosis), 'gpt-5-mini')
        
    # User Centric Diagnosis
    user_feedback_dict = {}
    rec_pairs = []
    with torch.no_grad():
        with open(f'{args.dataset}/user_sets.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        all_candi = [jj for jj in range(1, itemnum+1)]
        sampled_users = random.sample(lines, 20)
        for s_u in tqdm(sampled_users):
            s_uu = s_u.strip()
            hist = user_train['History'][s_uu]
            length_idx = args.maxlen - 1
            candi = list(set(all_candi)-set(hist))
            seq = np.zeros([args.maxlen], dtype=np.int32)
            for i in reversed(hist):
                seq[length_idx] = i
                length_idx -=1
            uuu, seq, candi = np.array([int(s_uu)]), np.array([seq]), np.array([candi])
            logits = model(uuu, seq, item_indices=candi)
            top_values, top_indices = torch.topk(logits, 20, dim=1)
            rec_results = [int(candi[0][t]) for t in top_indices[0].detach().cpu().tolist()]
            rec_pairs.append((s_uu, rec_results))
                        
    max_workers = min(4, len(rec_pairs))
    def _run_simulation(user_id, rec_results):
        return user_id, simulator.conduct_simulation(user_id, rec_results)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_simulation, u, rec) for u, rec in rec_pairs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Simulating (parallel)"):
            try:
                u, feedback = fut.result()
                user_feedback_dict[u] = feedback
            except Exception as e:
                print(f"[Simulation failed] {e}")
                
    critical_indication, simulator_prompt, diagnosis_prompt = get_summarized_suggestion(user_feedback_dict, math_diagnosis,math_results,'gpt-5-mini', aggregator_prompt)
    
    if critical_indication:
        return 1, {
        "ndcg_score(0-1)": 0,
        "hr_score(0-1)": 0,
        "combined_score":(0.6*0 + 0.4*0),
        "simulator_comment": simulator_prompt,
        "diagnosis_comment": diagnosis_prompt
        }
    
    return 1, {
        "ndcg_score(0-1)": best_valid_ndcg,
        "hr_score(0-1)": best_valid_hr,
        "combined_score":(0.6*best_valid_hr + 0.4*best_valid_ndcg),
        "simulator_comment": simulator_prompt,
        "diagnosis_comment": diagnosis_prompt
        }


if __name__ =='__main__':
    args = get_args()
    results = main(args)
    