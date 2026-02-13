### >>> Self_EvolveRec-BLOCK-START: Add warnings import for diagnostics error reporting
import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter
import warnings

### <<< Self_EvolveRec-BLOCK-END


def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
    You are a Senior AI Researcher analyzing the mathematical health of a Recommendation System.
    
    Here are the raw mathematical metrics measured from the current model:
    {json.dumps(raw_diagnosis['metrics'], indent=2)}
    
    Here are the definitions of what these metrics mean:
    {json.dumps(raw_diagnosis['metric_definitions'], indent=2)}
    
    Your task: produce a concise diagnosis summary that captures ONLY the core findings and implications.
    Do NOT propose web searches or action plans. Do NOT list "what to look up".

    Strictly follow the JSON format below.

    Output Format (JSON):
    {{
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE>,
    "core_findings": [
        "<1-2 sentences: the most important interpretation of the metrics>",
        "<1-2 sentences: the second most important interpretation (only if truly necessary)>"
    ],
    "key_implications": [
        "<1 sentence: what this implies about model behavior>",
        "<1 sentence: what this implies about training dynamics or representation>",
        "<optional 1 sentence: risk/trade-off if relevant>"
    ],
    "evidence": {{
        "headline_metrics": {{
        "<metric_name>": <value>,
        "<metric_name>": <value>
        }},
        "brief_metric_read": {{
        "<metric_name>": "<very short interpretation tied to its definition>",
        "<metric_name>": "<very short interpretation tied to its definition>"
        }}
    }}
    }}

    Rules:
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    def __init__(self, model, device):
        self.model = model
        # DEBUG: device was a str in some runs, causing AttributeError when accessing device.type.
        # Normalize to torch.device and align with the model's device to avoid device mismatches.
        try:
            model_device = getattr(model, "dev", None)
            if isinstance(model_device, torch.device):
                self.device = model_device
            elif isinstance(model_device, str):
                self.device = torch.device(model_device)
            elif isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_data = model.meta_data
        self.review_data = model.review_data
        self.user_train = model.user_train

    def _get_categories(self, item_id: int):
        key = str(int(item_id))
        info = self.meta_data.get(key, None)
        if not isinstance(info, dict):
            return []
        cats = info.get("categories", [])
        if isinstance(cats, list):
            return [c for c in cats if isinstance(c, str)]
        return []

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]  # drop padding idx=0

            n = emb.size(0)
            if n <= 2:
                return 0.0

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            gram = x @ x.t()
            offdiag_sum = gram.sum() - gram.diag().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        self.model.eval()

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            seq_np = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq
            log_feats = self.model.log2feats(seq_np)

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            valid = pos_t != 0
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = pos_cpu != 0

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            for b, l in np.argwhere(low_mask):
                low_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    low_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    low_neg_cat[c] += 1

            for b, l in np.argwhere(wrong_mask):
                wrong_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    wrong_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    wrong_neg_cat[c] += 1

        if len(margins_all) == 0:
            margin_metrics = {
                "mean": 0.0,
                "p10": 0.0,
                "neg_beats_pos_rate": 0.0,
                "small_margin_rate": 0.0,
            }
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        breakdown = {
            "eps": float(eps),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_top_categories": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_top_categories": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
        }

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Adaptive AtP verification: AtP-lite with bootstrap CIs and selective activation-patch calibration; pooled inference, bias/resurfacing probes
    def run_full_diagnosis(self, train_loader):
        """
        Adaptive AtP verification diagnosis suite (GPU-friendly).
        Returns a machine-readable MODEL_DIAGNOSIS dict with metrics, CIs, subgroup breakdowns, and AtP calibration.
        """
        import contextlib

        device = self.device
        model = self.model
        model.eval()

        # ----------------------------
        # Helpers
        # ----------------------------
        def _ensure_cat_cache():
            if hasattr(self, "_cat2items_cache"):
                return
            item2cat = model.item2cat.detach().cpu().numpy()
            cat2items = {}
            for i in range(1, model.item_num + 1):
                c = int(item2cat[i])
                if c not in cat2items:
                    cat2items[c] = []
                cat2items[c].append(i)
            self._cat2items_cache = {
                c: np.asarray(v, dtype=np.int64) for c, v in cat2items.items()
            }

        def _pearson_corr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
            x = x.float()
            y = y.float()
            xm, ym = x.mean(), y.mean()
            xv, yv = x - xm, y - ym
            denom = (xv.square().mean().sqrt() * yv.square().mean().sqrt()).clamp_min(
                eps
            )
            return float(((xv * yv).mean() / denom).item())

        @torch.no_grad()
        def _compute_logits_for_candidates(
            seq_np: np.ndarray, candidates_2d: torch.Tensor
        ):
            """
            Inference-consistent rescoring using model.log2feats + pool_user + predict_score.
            """
            amp_ctx = (
                torch.cuda.amp.autocast()
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with amp_ctx:
                log_feats = model.log2feats(seq_np)  # [B, L, H]
                nonpad_mask = torch.as_tensor(
                    seq_np, dtype=torch.long, device=device
                ).ne(0)
                pooled = model.pool_user(log_feats, nonpad_mask)  # [B, H]
                final_feat = pooled.unsqueeze(1)  # [B, 1, H]
                candi3 = candidates_2d.unsqueeze(1)  # [B, 1, C]
                logits = model.predict_score(final_feat, candi3).squeeze(1)  # [B, C]
            return logits

        def _last_nonzero_pos(row: np.ndarray) -> int:
            nz = np.nonzero(row)[0]
            return int(nz[-1]) if len(nz) > 0 else -1

        def _second_last_nonzero_pos(row: np.ndarray) -> int:
            nz = np.nonzero(row)[0]
            return int(nz[-2]) if len(nz) > 1 else -1

        def _build_edits(seq_np: np.ndarray):
            """
            Minimal plausibility edits: mask last; swap last two; duplicate last.
            """
            B, L = seq_np.shape
            edits = {}
            # mask last non-zero
            mask_arr = seq_np.copy()
            for i in range(B):
                j = _last_nonzero_pos(mask_arr[i])
                if j >= 0:
                    mask_arr[i, j] = 0
            edits["mask"] = mask_arr
            # swap last two non-zeros
            swap_arr = seq_np.copy()
            for i in range(B):
                j1 = _last_nonzero_pos(swap_arr[i])
                j2 = _second_last_nonzero_pos(swap_arr[i])
                if j1 >= 0 and j2 >= 0 and j1 != j2:
                    tmp = swap_arr[i, j1]
                    swap_arr[i, j1] = swap_arr[i, j2]
                    swap_arr[i, j2] = tmp
            edits["swap"] = swap_arr
            # duplicate: replace second-last with last (if exists)
            dup_arr = seq_np.copy()
            for i in range(B):
                j1 = _last_nonzero_pos(dup_arr[i])
                j2 = _second_last_nonzero_pos(dup_arr[i])
                if j1 >= 0 and j2 >= 0:
                    dup_arr[i, j2] = dup_arr[i, j1]
            edits["dup"] = dup_arr
            return edits

        def _batch_topk_ids(
            candidates_2d: torch.Tensor, logits_2d: torch.Tensor, k: int
        ):
            k = max(1, int(k))
            _, topi = torch.topk(logits_2d, k=k, dim=1)
            return candidates_2d.gather(1, topi)

        def _batch_jaccard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if a.size(1) == 0 or b.size(1) == 0:
                return torch.zeros(a.size(0), device=a.device, dtype=torch.float32)
            eq = a.unsqueeze(2) == b.unsqueeze(1)  # [B, K, K]
            inter = eq.any(dim=2).sum(dim=1).float()
            K = a.size(1)
            union = (2 * K - inter).clamp_min(1.0)
            return inter / union

        def _rbo_row(a: np.ndarray, b: np.ndarray, p: float = 0.9) -> float:
            K = min(len(a), len(b))
            if K == 0:
                return 0.0
            a_seen, b_seen = set(), set()
            rbo = 0.0
            for d in range(1, K + 1):
                a_seen.add(int(a[d - 1]))
                b_seen.add(int(b[d - 1]))
                overlap = len(a_seen.intersection(b_seen))
                rbo += (overlap / d) * (p ** (d - 1))
            return float((1 - p) * rbo)

        def _batch_rbo(
            a_ids: torch.Tensor, b_ids: torch.Tensor, p: float = 0.9
        ) -> float:
            a_np = a_ids.detach().cpu().numpy()
            b_np = b_ids.detach().cpu().numpy()
            vals = [_rbo_row(a_np[i], b_np[i], p=p) for i in range(a_np.shape[0])]
            return float(np.mean(vals)) if len(vals) > 0 else 0.0

        def _get_recent_categories(seq_np: np.ndarray, last_k: int = 5):
            item2cat_cpu = model.item2cat.detach().cpu().numpy()
            cats = []
            for row in seq_np:
                nz = np.array([i for i, v in enumerate(row) if v != 0], dtype=np.int64)[
                    -last_k:
                ]
                items = row[nz] if nz.size > 0 else np.array([], dtype=np.int64)
                catset = set(int(item2cat_cpu[int(it)]) for it in items if int(it) > 0)
                cats.append(catset)
            return cats

        def _get_user_disliked_sets(user_ids_batch):
            disliked = []
            for uid in user_ids_batch:
                key = str(int(uid))
                u = self.review_data.get(key, {})
                items = set()
                if isinstance(u, dict):
                    for item_id_str, r in u.items():
                        try:
                            rating = float(r.get("rating", 0.0))
                            if rating <= 2.0:
                                items.add(int(item_id_str))
                        except Exception:
                            continue
                disliked.append(items)
            return disliked

        def _build_candidates_for_batch(
            seq_np: np.ndarray, C: int = 256, pop_frac=0.5, cat_frac=0.4
        ):
            """
            Mixed candidate pool: popularity head + recent-category + niche tail (CPU build, GPU scoring).
            """
            _ensure_cat_cache()
            B = seq_np.shape[0]
            C_pop = max(1, int(C * pop_frac))
            C_cat = max(0, int(C * cat_frac))
            C_niche = max(0, C - C_pop - C_cat)

            pop_norm = model.pop_norm.detach().cpu()
            pop_norm[0] = -1.0
            top_pop_vals, top_pop_idx = torch.topk(pop_norm, k=min(C, model.item_num))
            pop_sorted = top_pop_idx[top_pop_vals >= 0].cpu().numpy()
            niche_pool = pop_sorted[::-1]
            item2cat_cpu = model.item2cat.detach().cpu().numpy()

            cands = np.zeros((B, C), dtype=np.int64)
            rng = np.random.default_rng(12345)

            for i in range(B):
                row = seq_np[i]
                nz = row[row != 0]
                recent_cat = int(item2cat_cpu[int(nz[-1])]) if nz.size > 0 else 0
                cat_pool = self._cat2items_cache.get(recent_cat, pop_sorted)

                part_pop = pop_sorted[: min(C_pop, pop_sorted.shape[0])]
                if part_pop.shape[0] > 0 and part_pop.shape[0] < C_pop:
                    pop_sample = rng.choice(
                        part_pop, size=part_pop.shape[0], replace=False
                    )
                elif part_pop.shape[0] >= C_pop:
                    pop_sample = rng.choice(part_pop, size=C_pop, replace=False)
                else:
                    pop_sample = np.array([], dtype=np.int64)

                if cat_pool.shape[0] > 0:
                    cat_take = min(C_cat, cat_pool.shape[0])
                    cat_sample = rng.choice(cat_pool, size=cat_take, replace=False)
                else:
                    cat_sample = np.array([], dtype=np.int64)

                if niche_pool.shape[0] > 0:
                    niche_take = min(C_niche, niche_pool.shape[0])
                    niche_sample = rng.choice(
                        niche_pool, size=niche_take, replace=False
                    )
                else:
                    niche_sample = np.array([], dtype=np.int64)

                pool = np.unique(
                    np.concatenate([pop_sample, cat_sample, niche_sample], axis=0)
                )
                pool = pool[pool > 0]

                if pool.shape[0] < C:
                    pad_need = C - pool.shape[0]
                    head_pad = pop_sorted[: min(pad_need, pop_sorted.shape[0])]
                    if head_pad.shape[0] > 0:
                        pad_sample = rng.choice(
                            head_pad,
                            size=min(pad_need, head_pad.shape[0]),
                            replace=False,
                        )
                        pool = np.unique(np.concatenate([pool, pad_sample], axis=0))

                pool = pool[:C] if pool.shape[0] >= C else np.resize(pool, C)
                cands[i] = pool

            return cands

        def _bootstrap_ci(vals: list, n_resamples: int = 100, alpha: float = 0.05):
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                return (0.0, 0.0)
            rng = np.random.default_rng(2025)
            means = []
            for _ in range(n_resamples):
                idx = rng.integers(0, arr.size, size=arr.size)
                means.append(arr[idx].mean())
            lo = float(np.percentile(means, 100 * (alpha / 2)))
            hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
            return (lo, hi)

        # Capture last attention outputs and optionally per-head weights; allow overriding last attn output
        def _logfeats_with_capture(
            seq_t: torch.Tensor,
            need_head_weights: bool = False,
            override_last_attn_out: torch.Tensor = None,
        ):
            model.eval()
            with torch.enable_grad():
                log_seqs_t = seq_t  # [B,L]
                seqs = model.get_item_rep(log_seqs_t) * (
                    model.item_emb.embedding_dim**0.5
                )
                B, L = log_seqs_t.shape
                pos_ids = (
                    torch.arange(1, L + 1, device=device).unsqueeze(0).expand(B, -1)
                )
                mask_nonzero = log_seqs_t.ne(0).long()
                pos_ids = pos_ids * mask_nonzero
                seqs = model.emb_dropout(seqs + model.pos_emb(pos_ids))
                seqs0 = seqs
                tl = seqs.shape[1]
                attention_mask = model.causal_mask[:tl, :tl]
                key_pad_mask = log_seqs_t.eq(0)
                captured_weights = None
                last_attn_out = None
                for i in range(len(model.attention_layers)):
                    seqs = torch.transpose(seqs, 0, 1)  # [L,B,H]
                    if i == len(model.attention_layers) - 1:
                        attn = model.attention_layers[i]
                        mha_outputs, attn_w = attn(
                            seqs,
                            seqs,
                            seqs,
                            attn_mask=attention_mask,
                            key_padding_mask=key_pad_mask,
                            need_weights=True,
                            average_attn_weights=not need_head_weights,
                        )
                        # Normalize attn_w to [B, H, L, L] if needed
                        if need_head_weights:
                            if isinstance(attn_w, torch.Tensor) and attn_w.dim() == 4:
                                captured_weights = attn_w  # [B, num_heads, L, L]
                            else:
                                try:
                                    # Fallback from (B*num_heads, L, L)
                                    Hh = (
                                        attn.num_heads
                                        if hasattr(attn, "num_heads")
                                        else 2
                                    )
                                    Bh = int(attn_w.size(0) // Hh)
                                    captured_weights = attn_w.view(Bh, Hh, tl, tl)
                                except Exception:
                                    captured_weights = None
                        last_attn_out = mha_outputs
                        use_out = (
                            override_last_attn_out
                            if override_last_attn_out is not None
                            else mha_outputs
                        )
                        seqs = model.attention_layernorms[i](seqs + use_out)
                    else:
                        mha_outputs, _ = model.attention_layers[i](
                            seqs,
                            seqs,
                            seqs,
                            attn_mask=attention_mask,
                            key_padding_mask=key_pad_mask,
                        )
                        seqs = model.attention_layernorms[i](seqs + mha_outputs)
                    seqs = torch.transpose(seqs, 0, 1)  # [B,L,H]
                    seqs = model.forward_layernorms[i](
                        seqs + model.forward_layers[i](seqs)
                    )
                log_feats = model.last_layernorm(seqs)
                return log_feats, last_attn_out, captured_weights, seqs0

        # ----------------------------
        # Base metrics: embedding collapse and margin breakdown
        # ----------------------------
        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )

        # ----------------------------
        # Probes over a few batches
        # ----------------------------
        K_TOP = 20
        C_CAND = 256
        NUM_PROBE_BATCHES = 5
        RBO_P = 0.9
        np.random.seed(1337)
        random.seed(1337)
        torch.manual_seed(1337)

        # Global refs
        global_pop_mean = (
            float(model.pop_norm[1:].mean().item())
            if model.pop_norm.numel() > 1
            else 0.0
        )
        global_price_mean = (
            float(model.item2price[1:].float().mean().item())
            if model.item2price.numel() > 1
            else 0.0
        )

        # Popularity tertiles for subgroup breakdowns
        pop_vals = model.pop_norm[1:].detach().cpu().numpy()
        if pop_vals.size > 0:
            t1, t2 = np.quantile(pop_vals, [1 / 3, 2 / 3])
        else:
            t1, t2 = 0.0, 0.0

        seq_sens_mask, seq_sens_swap, seq_sens_dup = [], [], []
        rbo_mask, rbo_swap, rbo_dup = [], [], []
        delta_top1_mask, delta_top1_swap, delta_top1_dup = [], [], []

        pop_corr_pool, price_corr_pool = [], []
        topk_pop_vals, topk_price_vals = [], []
        subgroup_fracs = {"head": [], "mid": [], "tail": []}

        off_genre_hits = 0
        off_genre_total = 0
        repetition_hits = 0
        repetition_total = 0
        disliked_hits = 0
        disliked_total = 0

        with torch.no_grad():
            for step, batch in enumerate(train_loader):
                if step >= NUM_PROBE_BATCHES:
                    break
                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    continue

                user_ids = batch[0]
                seq = batch[1]
                seq_np = (
                    seq.detach().cpu().numpy()
                    if isinstance(seq, torch.Tensor)
                    else np.asarray(seq, dtype=np.int64)
                )
                B, L = seq_np.shape

                # Candidates
                cands_np = _build_candidates_for_batch(seq_np, C=C_CAND)
                cands = torch.as_tensor(cands_np, dtype=torch.long, device=device)

                # Base logits and topK
                base_logits = _compute_logits_for_candidates(seq_np, cands)
                K_eff = int(min(K_TOP, base_logits.size(1)))
                base_top_ids = _batch_topk_ids(cands, base_logits, k=K_eff)
                base_top1_idx = base_logits.argmax(dim=1)
                base_top1_item = cands.gather(1, base_top1_idx.unsqueeze(1)).squeeze(1)

                # Bias probes
                pop_norm_cands = model.pop_norm[cands]
                price_bins_cands = model.item2price[cands].float()
                pop_corr_pool.append(
                    _pearson_corr_torch(base_logits.flatten(), pop_norm_cands.flatten())
                )
                price_corr_pool.append(
                    _pearson_corr_torch(
                        base_logits.flatten(), price_bins_cands.flatten()
                    )
                )
                tk_idx = base_logits.topk(k=K_eff, dim=1).indices
                topk_pop_vals.append(
                    float(pop_norm_cands.gather(1, tk_idx).float().mean().item())
                )
                topk_price_vals.append(
                    float(price_bins_cands.gather(1, tk_idx).float().mean().item())
                )

                # Subgroup fractions by popularity tertiles
                topk_ids_np = base_top_ids.detach().cpu().numpy()
                for i in range(B):
                    ids = topk_ids_np[i]
                    pops = (
                        model.pop_norm[torch.as_tensor(ids, device=device)]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    if pops.size == 0:
                        continue
                    head = np.mean(pops >= t2)
                    tail = np.mean(pops < t1)
                    mid = 1.0 - head - tail
                    subgroup_fracs["head"].append(float(head))
                    subgroup_fracs["mid"].append(float(mid))
                    subgroup_fracs["tail"].append(float(tail))

                # Off-genre rate
                recent_cat_sets = _get_recent_categories(seq_np, last_k=5)
                topk_cat_ids = model.item2cat[base_top_ids].detach().cpu().numpy()
                for i in range(B):
                    if len(recent_cat_sets[i]) == 0:
                        continue
                    off = sum(
                        1
                        for k in range(K_eff)
                        if int(topk_cat_ids[i, k]) not in recent_cat_sets[i]
                    )
                    off_genre_hits += off
                    off_genre_total += K_eff

                # Repetition rate
                for i in range(B):
                    hist_set = set(int(x) for x in seq_np[i] if int(x) > 0)
                    if len(hist_set) == 0:
                        continue
                    rec_items = base_top_ids[i].detach().cpu().numpy().tolist()
                    repetition_hits += sum(1 for it in rec_items if int(it) in hist_set)
                    repetition_total += len(rec_items)

                # Disliked resurfacing
                if isinstance(user_ids, torch.Tensor):
                    uids_list = user_ids.detach().cpu().numpy().tolist()
                elif isinstance(user_ids, (list, tuple)):
                    uids_list = [int(u) for u in user_ids]
                else:
                    try:
                        uids_list = list(user_ids)
                    except Exception:
                        uids_list = [0] * B
                disliked_sets = _get_user_disliked_sets(uids_list)
                for i in range(B):
                    if len(disliked_sets[i]) == 0:
                        continue
                    rec_items = base_top_ids[i].detach().cpu().numpy().tolist()
                    disliked_hits += sum(
                        1 for it in rec_items if int(it) in disliked_sets[i]
                    )
                    disliked_total += len(rec_items)

                # Edits
                edits = _build_edits(seq_np)
                for edit_name, edit_arr in edits.items():
                    edit_logits = _compute_logits_for_candidates(edit_arr, cands)
                    edit_top_ids = _batch_topk_ids(cands, edit_logits, k=K_eff)
                    jacc = _batch_jaccard(base_top_ids, edit_top_ids)
                    sens = (1.0 - jacc).mean().item()
                    # Δlogit for base top1 item
                    base_item = base_top1_item
                    base_item_col = (cands == base_item.unsqueeze(1)).float()
                    base_item_idx = base_item_col.argmax(dim=1)
                    base_top1_logit = base_logits.gather(
                        1, base_item_idx.unsqueeze(1)
                    ).squeeze(1)
                    edit_top1_logit = edit_logits.gather(
                        1, base_item_idx.unsqueeze(1)
                    ).squeeze(1)
                    delta = (edit_top1_logit - base_top1_logit).abs().mean().item()
                    # RBO
                    rbo_val = _batch_rbo(base_top_ids, edit_top_ids, p=RBO_P)

                    if edit_name == "mask":
                        seq_sens_mask.append(sens)
                        delta_top1_mask.append(delta)
                        rbo_mask.append(rbo_val)
                    elif edit_name == "swap":
                        seq_sens_swap.append(sens)
                        delta_top1_swap.append(delta)
                        rbo_swap.append(rbo_val)
                    elif edit_name == "dup":
                        seq_sens_dup.append(sens)
                        delta_top1_dup.append(delta)
                        rbo_dup.append(rbo_val)

        # Aggregate pool metrics
        def _mean_or_zero(lst):
            return float(np.mean(lst)) if len(lst) > 0 else 0.0

        seq_sensitivity_mask = _mean_or_zero(seq_sens_mask)
        seq_sensitivity_swap = _mean_or_zero(seq_sens_swap)
        seq_sensitivity_dup = _mean_or_zero(seq_sens_dup)
        delta_top1_mask_mean = _mean_or_zero(delta_top1_mask)
        delta_top1_swap_mean = _mean_or_zero(delta_top1_swap)
        delta_top1_dup_mean = _mean_or_zero(delta_top1_dup)
        pop_bias_corr = _mean_or_zero(pop_corr_pool)
        price_bias_corr = _mean_or_zero(price_corr_pool)
        topk_popularity_mean = _mean_or_zero(topk_pop_vals)
        topk_price_mean = _mean_or_zero(topk_price_vals)
        off_genre_rate = float(off_genre_hits / max(1, off_genre_total))
        repetition_rate = float(repetition_hits / max(1, repetition_total))
        disliked_resurfacing_rate = (
            float(disliked_hits / max(1, disliked_total)) if disliked_total > 0 else 0.0
        )
        rbo_mask_mean = _mean_or_zero(rbo_mask)
        rbo_swap_mean = _mean_or_zero(rbo_swap)
        rbo_dup_mean = _mean_or_zero(rbo_dup)
        topk_pop_split = {
            "head": _mean_or_zero(subgroup_fracs["head"]),
            "mid": _mean_or_zero(subgroup_fracs["mid"]),
            "tail": _mean_or_zero(subgroup_fracs["tail"]),
        }

        # Bootstrap CIs for selected metrics
        ci = {
            "seq_sensitivity_mask_CI": _bootstrap_ci(seq_sens_mask, n_resamples=100),
            "seq_sensitivity_swap_CI": _bootstrap_ci(seq_sens_swap, n_resamples=100),
            "seq_sensitivity_duplication_CI": _bootstrap_ci(
                seq_sens_dup, n_resamples=100
            ),
            "pop_bias_corr_pool_CI": _bootstrap_ci(pop_corr_pool, n_resamples=100),
        }

        # ----------------------------
        # Adaptive AtP-lite + selective activation patch calibration
        # ----------------------------
        attribution_recency_importance_ratio = 0.0
        activation_recovery_last_layer = 0.0
        head_recency_weight_top = 0.0
        embedding_contribution_to_topk = 0.0
        atp_validated = False
        patch_budget_used = 0
        calib_coef = 0.0
        calib_intercept = 0.0
        calib_r2 = 0.0

        # Escalation criterion: wide CI on seq_sensitivity_mask
        ci_width = ci["seq_sensitivity_mask_CI"][1] - ci["seq_sensitivity_mask_CI"][0]
        NEED_VERIFY = bool(ci_width > 0.25)

        try:
            for step, batch in enumerate(train_loader):
                if step > 0:
                    break  # single small batch for AtP
                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    continue

                seq = batch[1]
                seq_np = (
                    seq.detach().cpu().numpy()
                    if isinstance(seq, torch.Tensor)
                    else np.asarray(seq, dtype=np.int64)
                )
                B_all, L = seq_np.shape
                B = min(B_all, 16)
                seq_np = seq_np[:B]

                C_ATT = 128
                cands_np = _build_candidates_for_batch(seq_np, C=C_ATT)
                cands = torch.as_tensor(cands_np, dtype=torch.long, device=device)

                seq_t_clean = torch.as_tensor(seq_np, dtype=torch.long, device=device)
                seq_t_corrupt = torch.as_tensor(
                    _build_edits(seq_np)["mask"], dtype=torch.long, device=device
                )

                with torch.enable_grad():
                    log_feats_clean, last_attn_clean, head_w_clean, seqs0_clean = (
                        _logfeats_with_capture(
                            seq_t_clean,
                            need_head_weights=True,
                            override_last_attn_out=None,
                        )
                    )
                    nonpad_mask_clean = seq_t_clean.ne(0)
                    pooled_clean = model.pool_user(log_feats_clean, nonpad_mask_clean)
                    logits_clean = model.predict_score(
                        pooled_clean.unsqueeze(1), cands.unsqueeze(1)
                    ).squeeze(1)
                    K_eff = int(min(K_TOP, logits_clean.size(1)))
                    base_top1_idx = logits_clean.argmax(dim=1)
                    base_top1_item = cands.gather(
                        1, base_top1_idx.unsqueeze(1)
                    ).squeeze(1)
                    # Head recency weight: max per-head weight on last->last
                    if head_w_clean is not None:
                        rec_w = []
                        for i in range(B):
                            nz = (
                                (seq_t_clean[i] != 0)
                                .nonzero(as_tuple=False)
                                .squeeze(-1)
                            )
                            if nz.numel() == 0:
                                continue
                            li = int(nz[-1].item())
                            wi = (
                                head_w_clean[i, :, li, li]
                                if head_w_clean.dim() == 4
                                else None
                            )
                            if wi is None:
                                continue
                            rec_w.append(wi)
                        if len(rec_w) > 0:
                            rec_w = torch.stack(rec_w, dim=0).mean(dim=0)  # [H]
                            head_recency_weight_top = float(rec_w.max().item())

                with torch.enable_grad():
                    log_feats_cor, last_attn_cor, _, seqs0_cor = _logfeats_with_capture(
                        seq_t_corrupt,
                        need_head_weights=False,
                        override_last_attn_out=None,
                    )
                    nonpad_mask_cor = seq_t_corrupt.ne(0)
                    pooled_cor = model.pool_user(log_feats_cor, nonpad_mask_cor)
                    logits_cor = model.predict_score(
                        pooled_cor.unsqueeze(1), cands.unsqueeze(1)
                    ).squeeze(1)

                    match_mat = (cands == base_top1_item.unsqueeze(1)).float()
                    base_item_idx = match_mat.argmax(dim=1)
                    target_scalar = logits_cor.gather(
                        1, base_item_idx.unsqueeze(1)
                    ).sum()
                    grad_last = torch.autograd.grad(
                        target_scalar,
                        last_attn_cor,
                        retain_graph=False,
                        allow_unused=True,
                    )[0]
                    if grad_last is None:
                        grad_last = torch.zeros_like(last_attn_cor)

                # AtP-lite token importance
                diff_attn = last_attn_clean.detach() - last_attn_cor.detach()  # [L,B,H]
                token_imp = (diff_attn * grad_last).sum(dim=2).abs()  # [L,B]
                valid = seq_t_clean.ne(0).float()
                token_imp_bt = token_imp.transpose(0, 1) * valid  # [B,L]
                total_imp = token_imp_bt.sum(dim=1).clamp_min(1e-6)
                last_pos_mask = (seq_t_clean != 0).cumsum(dim=1)
                last_pos_mask = (
                    last_pos_mask == last_pos_mask.max(dim=1, keepdim=True).values
                ).float()
                last_imp = (token_imp_bt * last_pos_mask).sum(dim=1)
                attribution_recency_importance_ratio = float(
                    (last_imp / total_imp).mean().item()
                )

                # Activation patching: validate & measure recovery
                with torch.no_grad():
                    log_feats_patched, _, _, _ = _logfeats_with_capture(
                        seq_t_corrupt,
                        need_head_weights=False,
                        override_last_attn_out=last_attn_clean.detach(),
                    )
                    pooled_patched = model.pool_user(log_feats_patched, nonpad_mask_cor)
                    logits_patched = model.predict_score(
                        pooled_patched.unsqueeze(1), cands.unsqueeze(1)
                    ).squeeze(1)
                    base_top_ids = _batch_topk_ids(cands, logits_clean, k=K_eff)
                    corrupt_top_ids = _batch_topk_ids(cands, logits_cor, k=K_eff)
                    patched_top_ids = _batch_topk_ids(cands, logits_patched, k=K_eff)
                    j_base_cor = _batch_jaccard(base_top_ids, corrupt_top_ids).mean()
                    j_base_pat = _batch_jaccard(base_top_ids, patched_top_ids).mean()
                    activation_recovery_last_layer = float(
                        (j_base_pat - j_base_cor).clamp(min=0.0).item()
                    )

                # Grad x input contribution on initial fused embeddings to base top-1
                with torch.enable_grad():
                    log_feats_c2, _, _, seqs0_c2 = _logfeats_with_capture(
                        seq_t_clean, need_head_weights=False
                    )
                    pooled_c2 = model.pool_user(log_feats_c2, nonpad_mask_clean)
                    logits_c2 = model.predict_score(
                        pooled_c2.unsqueeze(1), cands.unsqueeze(1)
                    ).squeeze(1)
                    base_top1_idx2 = logits_c2.argmax(dim=1)
                    base_top1_item2 = cands.gather(
                        1, base_top1_idx2.unsqueeze(1)
                    ).squeeze(1)
                    match_mat2 = (cands == base_top1_item2.unsqueeze(1)).float()
                    base_item_idx2 = match_mat2.argmax(dim=1)
                    target2 = logits_c2.gather(1, base_item_idx2.unsqueeze(1)).sum()
                    grads_seq0 = torch.autograd.grad(
                        target2, seqs0_c2, retain_graph=False, allow_unused=True
                    )[0]
                    if grads_seq0 is None:
                        grads_seq0 = torch.zeros_like(seqs0_c2)
                    cont_map = (grads_seq0 * seqs0_c2).abs().sum(
                        dim=2
                    ) * nonpad_mask_clean.float()  # [B,L]
                    frac_tail3 = []
                    for i in range(B):
                        nz = (seq_t_clean[i] != 0).nonzero(as_tuple=False).squeeze(-1)
                        if nz.numel() == 0:
                            continue
                        li = int(nz[-1].item())
                        st = max(0, li - 2)
                        num = cont_map[i, st : li + 1].sum()
                        den = cont_map[i].sum().clamp_min(1e-6)
                        frac_tail3.append((num / den).item())
                    embedding_contribution_to_topk = (
                        float(np.mean(frac_tail3)) if len(frac_tail3) > 0 else 0.0
                    )

                # Selective calibration across samples (small budget)
                # AtP predicted effect per-sample = last_imp/total_imp; exact effect = Jaccard recovery per-sample
                if NEED_VERIFY:
                    # Compute per-sample exact recovery
                    with torch.no_grad():
                        base_top_ids_b = _batch_topk_ids(cands, logits_clean, k=K_eff)
                        corrupt_top_ids_b = _batch_topk_ids(cands, logits_cor, k=K_eff)
                        patched_top_ids_b = _batch_topk_ids(
                            cands, logits_patched, k=K_eff
                        )

                        # Per-row Jaccards
                        def _rowwise_jacc(a, b):
                            eq = a.unsqueeze(2) == b.unsqueeze(1)
                            inter = eq.any(dim=2).sum(dim=1).float()
                            K = a.size(1)
                            union = (2 * K - inter).clamp_min(1.0)
                            return inter / union

                        j_cor = _rowwise_jacc(base_top_ids_b, corrupt_top_ids_b)
                        j_pat = _rowwise_jacc(base_top_ids_b, patched_top_ids_b)
                        exact_eff = (j_pat - j_cor).clamp_min(0.0)  # [B]
                    atp_eff = (last_imp / total_imp).detach()  # [B]

                    # Choose top uncertain samples: here simply take largest atp_eff to stress-test
                    order = torch.argsort(atp_eff, descending=True)
                    take = min(int(max(1, 0.05 * B)), 8)
                    sel = order[:take]
                    x = atp_eff[sel].float()
                    y = exact_eff[sel].float()
                    if x.numel() > 0:
                        # Ridge fit: y ~ a*x + b
                        X = torch.stack([x, torch.ones_like(x)], dim=1)  # [n,2]
                        lam = 1e-3
                        XtX = X.t() @ X + lam * torch.eye(2, device=X.device)
                        Xty = X.t() @ y
                        coef = torch.linalg.solve(XtX, Xty)
                        calib_coef = float(coef[0].item())
                        calib_intercept = float(coef[1].item())
                        y_hat = X @ coef
                        ss_res = float(((y - y_hat).pow(2)).sum().item())
                        ss_tot = float(((y - y.mean()).pow(2)).sum().item() + 1e-8)
                        calib_r2 = float(max(0.0, 1.0 - ss_res / ss_tot))
                        atp_validated = bool(
                            calib_r2 >= 0.3
                        )  # minimal sanity threshold
                        patch_budget_used = int(take)
                    else:
                        atp_validated = False
                        patch_budget_used = 0
                else:
                    atp_validated = bool(activation_recovery_last_layer >= 0.05)
                    patch_budget_used = 0
        except Exception:
            # Keep defaults on any failure
            pass

        # Composite recommended-action score
        risk_raw = (
            0.40 * max(0.0, seq_sensitivity_mask)
            + 0.25 * max(0.0, seq_sensitivity_swap)
            + 0.15 * max(0.0, seq_sensitivity_dup)
            + 0.30 * abs(pop_bias_corr)
            + 0.20 * abs(price_bias_corr)
            + 0.30 * repetition_rate
            + 0.20 * off_genre_rate
        )
        recommended_action_score = float(max(0.0, min(1.0, risk_raw)))

        # Compose MODEL_DIAGNOSIS
        diagnosis = {
            "metrics": {
                # Representation health
                "embedding_collapse_score": round(float(collapse_score), 4),
                # Margin health
                "pos_neg_margin_mean": margin_metrics.get("mean", 0.0),
                "pos_neg_margin_p10": margin_metrics.get("p10", 0.0),
                "pos_neg_neg_beats_pos_rate": margin_metrics.get(
                    "neg_beats_pos_rate", 0.0
                ),
                "pos_neg_small_margin_rate": margin_metrics.get(
                    "small_margin_rate", 0.0
                ),
                "pos_neg_margin_category_breakdown": breakdown,
                # Sequence sensitivity
                "seq_sensitivity_mask": round(seq_sensitivity_mask, 4),
                "seq_sensitivity_swap": round(seq_sensitivity_swap, 4),
                "seq_sensitivity_duplication": round(seq_sensitivity_dup, 4),
                "delta_top1_logit_mask": round(delta_top1_mask_mean, 4),
                "delta_top1_logit_swap": round(delta_top1_swap_mean, 4),
                "delta_top1_logit_duplication": round(delta_top1_dup_mean, 4),
                # Rank stability (RBO; higher=more stable)
                "rbo_mask": round(rbo_mask_mean, 4),
                "rbo_swap": round(rbo_swap_mean, 4),
                "rbo_duplication": round(rbo_dup_mean, 4),
                # Bias probes
                "pop_bias_corr_pool": round(pop_bias_corr, 4),
                "price_bias_corr_pool": round(price_bias_corr, 4),
                "topk_popularity_mean": round(topk_popularity_mean, 4),
                "global_popularity_mean": round(global_pop_mean, 4),
                "topk_price_mean": round(topk_price_mean, 4),
                "global_price_mean": round(global_price_mean, 4),
                # Failure behaviors
                "off_genre_rate": round(off_genre_rate, 4),
                "repetition_rate_topk": round(repetition_rate, 4),
                "disliked_resurfacing_rate": round(disliked_resurfacing_rate, 4),
                # Attribution/patching-lite
                "head_importance_for_recency": round(head_recency_weight_top, 4),
                "attribution_recency_importance_ratio": round(
                    attribution_recency_importance_ratio, 4
                ),
                "activation_recovery_last_layer": round(
                    activation_recovery_last_layer, 4
                ),
                "embedding_contribution_to_topk": round(
                    embedding_contribution_to_topk, 4
                ),
                # Subgroups
                "topk_popularity_split_head": round(topk_pop_split["head"], 4),
                "topk_popularity_split_mid": round(topk_pop_split["mid"], 4),
                "topk_popularity_split_tail": round(topk_pop_split["tail"], 4),
                # Summary
                "recommended_action_score": round(recommended_action_score, 4),
            },
            "metric_definitions": {
                "embedding_collapse_score": "Range ~[0,1]. Mean pairwise cosine similarity of sampled item embeddings mapped to [0,1]. Higher = worse diversity.",
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher is better.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                "pos_neg_margin_category_breakdown": "For low-margin (margin<eps) and wrong-rank (margin<0) positions, counts meta_data categories for pos/neg items.",
                "seq_sensitivity_mask": "1 - Jaccard(base_topK, topK_after_mask_last). Higher = more fragile to masking recency.",
                "seq_sensitivity_swap": "1 - Jaccard(base_topK, topK_after_swap_last_two). Higher = order-sensitive.",
                "seq_sensitivity_duplication": "1 - Jaccard(base_topK, topK_after_duplicate_last).",
                "delta_top1_logit_mask": "Mean absolute change of the base-top1 item's logit after masking last item.",
                "delta_top1_logit_swap": "Mean absolute change after swapping last two items.",
                "delta_top1_logit_duplication": "Mean absolute change after duplicating the last item.",
                "rbo_mask": "Ranked-Biased Overlap (p=0.9) between base and masked topK. Higher = more stable.",
                "rbo_swap": "RBO (p=0.9) between base and swapped topK.",
                "rbo_duplication": "RBO (p=0.9) between base and duplicated topK.",
                "pop_bias_corr_pool": "Pearson correlation between candidate logits and item popularity (pop_norm).",
                "price_bias_corr_pool": "Pearson correlation between candidate logits and item price bins.",
                "topk_popularity_mean": "Mean popularity of recommended topK items.",
                "global_popularity_mean": "Mean popularity across catalog.",
                "topk_price_mean": "Mean price bucket of recommended topK items.",
                "global_price_mean": "Mean price bucket across catalog.",
                "off_genre_rate": "Fraction of topK outside user recent-5 categories.",
                "repetition_rate_topk": "Fraction of topK already seen in user history.",
                "disliked_resurfacing_rate": "Fraction of topK previously rated <= 2.0 by the user.",
                "head_importance_for_recency": "Max per-head attention weight at last position attending to itself (last layer).",
                "attribution_recency_importance_ratio": "AtP-lite fraction of token-importance mass at last position.",
                "activation_recovery_last_layer": "Jaccard recovery when patching corrupt run with clean last-attn outputs.",
                "embedding_contribution_to_topk": "Grad x input mass in last 3 tokens for base top-1 score.",
                "recommended_action_score": "Composite risk score [0,1] combining sensitivity, bias, repetition, and off-genre rates.",
            },
            "metrics_ci": {
                "seq_sensitivity_mask": [
                    round(ci["seq_sensitivity_mask_CI"][0], 4),
                    round(ci["seq_sensitivity_mask_CI"][1], 4),
                ],
                "seq_sensitivity_swap": [
                    round(ci["seq_sensitivity_swap_CI"][0], 4),
                    round(ci["seq_sensitivity_swap_CI"][1], 4),
                ],
                "seq_sensitivity_duplication": [
                    round(ci["seq_sensitivity_duplication_CI"][0], 4),
                    round(ci["seq_sensitivity_duplication_CI"][1], 4),
                ],
                "pop_bias_corr_pool": [
                    round(ci["pop_bias_corr_pool_CI"][0], 4),
                    round(ci["pop_bias_corr_pool_CI"][1], 4),
                ],
            },
            "subgroup_breakdowns": {
                "topk_popularity_split": {
                    "head": round(topk_pop_split["head"], 4),
                    "mid": round(topk_pop_split["mid"], 4),
                    "tail": round(topk_pop_split["tail"], 4),
                }
            },
            "verification_summary": {
                "ci_width_seq_sensitivity_mask": round(float(ci_width), 4),
                "atp_validated": bool(atp_validated),
                "patch_budget_used": int(patch_budget_used),
                "calibrator": {
                    "ridge_alpha": 1e-3,
                    "coef": round(float(calib_coef), 6),
                    "intercept": round(float(calib_intercept), 6),
                    "r2": round(float(calib_r2), 4),
                },
                "method": "Adaptive AtP-lite + bounded activation patch calibration",
            },
            "params": {
                "probe_num_batches": int(NUM_PROBE_BATCHES),
                "probe_candidates_per_user": int(C_CAND),
                "probe_topk": int(K_TOP),
                "eps_margin": float(eps),
                "device": str(device),
                "amp_autocast": bool(device.type == "cuda"),
                "bootstrap_n": 100,
                "rbo_p": float(RBO_P),
            },
        }

        return diagnosis


### <<< Self_EvolveRec-BLOCK-END
