"""
Advanced feature engineering for lottery ML models.

Extracts rich feature vectors from historical draw data:
- Frequency / Hot-Cold / Odd-Even / Sum / Span / AC / Consecutive
- Markov transition probabilities
- Poisson overdue scores
- Bayesian probabilities
- Sliding window statistics
- Time-series trends
- Pair / triplet co-occurrence features
"""
import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional


def build_features(
    df: pd.DataFrame, cfg,
    window_sizes: List[int] = None,
    seq_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrix X and label vector y for supervised ML.

    For each draw i (starting from min_history rows), produces a feature
    vector based on draws [0..i-1] and the label from draw i.

    Returns
    -------
    X : ndarray shape (n_samples, n_features)
    y : ndarray shape (n_samples, total_count)
    feature_names : list of str
    """
    cfgd = cfg  # compatibility
    main_nums = np.array([
        sorted([int(r[c]) for c in cfgd.main_cols])
        for _, r in df.iterrows()
    ])
    sub_nums = np.array([
        sorted([int(r[c]) for c in cfgd.sub_cols])
        for _, r in df.iterrows()
    ])
    return _build_feature_matrix(main_nums, sub_nums, cfgd, window_sizes, seq_length)


def _build_feature_matrix(
    main_nums: np.ndarray, sub_nums: np.ndarray, cfg,
    window_sizes: List[int] = None,
    seq_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Internal feature extraction."""
    if window_sizes is None:
        window_sizes = [20, 50, 100]

    n_draws = len(main_nums)
    total_count = cfg.main_count + cfg.sub_count
    main_range = np.arange(cfg.main_min, cfg.main_max + 1)
    sub_range = np.arange(cfg.sub_min, cfg.sub_max + 1)
    n_main = len(main_range)
    n_sub = len(sub_range)

    min_history = max(max(window_sizes), seq_length * 2)
    if n_draws < min_history + 10:
        return np.empty((0, 1)), np.empty((0, total_count)), ["none"]

    all_labels = np.concatenate([main_nums, sub_nums], axis=1)  # (n_draws, total_count)

    X_rows = []
    Y_rows = []
    feature_names = []

    # Precompute per-draw binary presence vectors
    main_presence = np.zeros((n_draws, n_main), dtype=np.float64)
    sub_presence = np.zeros((n_draws, n_sub), dtype=np.float64)
    for t in range(n_draws):
        for n in main_nums[t]:
            main_presence[t, n - cfg.main_min] = 1.0
        for n in sub_nums[t]:
            sub_presence[t, n - cfg.sub_min] = 1.0

    for t in range(min_history, n_draws):
        features = []

        # === 1. FREQUENCY (full history) ===
        main_full_freq = main_presence[:t].sum(axis=0) / max(t, 1)
        sub_full_freq = sub_presence[:t].sum(axis=0) / max(t, 1)
        features.extend(main_full_freq.tolist())
        features.extend(sub_full_freq.tolist())

        # === 2. SLIDING WINDOW FREQUENCIES ===
        for ws in window_sizes:
            w = min(ws, t)
            mw = main_presence[t - w:t].sum(axis=0) / w
            sw = sub_presence[t - w:t].sum(axis=0) / w
            features.extend(mw.tolist())
            features.extend(sw.tolist())

        # === 3. HOT/COLD COUNTS ===
        for ws in window_sizes:
            w = min(ws, t)
            mw = main_presence[t - w:t]
            sw = sub_presence[t - w:t]
            main_expected = w * cfg.main_count / n_main
            sub_expected = w * cfg.sub_count / n_sub
            hot_m = (mw.sum(axis=0) > main_expected * 1.3).sum()
            cold_m = (mw.sum(axis=0) < main_expected * 0.7).sum()
            hot_s = (sw.sum(axis=0) > sub_expected * 1.3).sum()
            cold_s = (sw.sum(axis=0) < sub_expected * 0.7).sum()
            features.extend([hot_m, cold_m, hot_s, cold_s])

        # === 4. ODD/EVEN RATIO (rolling) ===
        for ws in window_sizes:
            w = min(ws, t)
            window_mains = main_nums[t - w:t]
            odds = (window_mains % 2 == 1).sum(axis=1)
            features.append(odds.mean() / cfg.main_count)
            features.append(odds.std())
            if cfg.sub_count > 0:
                window_subs = sub_nums[t - w:t]
                s_odds = (window_subs % 2 == 1).sum(axis=1)
                features.append(s_odds.mean() / cfg.sub_count)
                features.append(s_odds.std())
            else:
                features.extend([0.5, 0.0])

        # === 5. SUM STATISTICS (rolling) ===
        for ws in window_sizes:
            w = min(ws, t)
            sums = main_nums[t - w:t].sum(axis=1)
            features.append(sums.mean())
            features.append(sums.std())
            if cfg.sub_count > 0:
                s_sums = sub_nums[t - w:t].sum(axis=1)
                features.append(s_sums.mean())
                features.append(s_sums.std())
            else:
                features.extend([0.0, 0.0])

        # === 6. SPAN STATISTICS (rolling) ===
        for ws in window_sizes:
            w = min(ws, t)
            spans = main_nums[t - w:t].max(axis=1) - main_nums[t - w:t].min(axis=1)
            features.append(spans.mean())
            features.append(spans.std())

        # === 7. AC VALUE STATS (rolling) ===
        for ws in window_sizes:
            w = min(ws, t)
            acs = np.array([_ac_value(row) for row in main_nums[t - w:t]])
            features.append(acs.mean())
            features.append(acs.std())

        # === 8. CONSECUTIVE STATS (rolling) ===
        for ws in window_sizes:
            w = min(ws, t)
            conss = np.array([_count_consecutive(row) for row in main_nums[t - w:t]])
            features.append(conss.mean())
            features.append(conss.std())

        # === 9. POISSON OVERDUE SCORES ===
        # Per number: probability it appears this draw given gap since last seen
        for arr, presence, n_range, n_min in [
            (main_nums[:t], main_presence[:t], n_main, cfg.main_min),
            (sub_nums[:t], sub_presence[:t], n_sub, cfg.sub_min),
        ]:
            for ni in range(n_range):
                num = ni + n_min
                last_seen = -1
                for tt in range(t - 1, -1, -1):
                    if presence[tt, ni] > 0:
                        last_seen = tt
                        break
                gap = t - 1 - last_seen if last_seen >= 0 else t + 1
                lam = max(presence[:, ni].sum() / max(t, 1) * t, 0.5)
                if lam > 0:
                    surv = math.exp(-gap / lam)
                    features.append(1.0 - surv)
                else:
                    features.append(1.0)

        # === 10. MARKOV TRANSITION PROBABILITIES ===
        # P(num appears | num appeared/absent last draw)
        for arr, presence, n_range, n_min in [
            (main_nums[:t], main_presence[:t], n_main, cfg.main_min),
            (sub_nums[:t], sub_presence[:t], n_sub, cfg.sub_min),
        ]:
            for ni in range(n_range):
                num = ni + n_min
                count_11 = 0  # appeared last AND appears now
                count_10 = 0  # appeared last AND absent now
                count_01 = 0  # absent last AND appears now
                for tt in range(1, t):
                    prev_present = presence[tt - 1, ni] > 0
                    curr_present = presence[tt, ni] > 0
                    if prev_present and curr_present:
                        count_11 += 1
                    elif prev_present and not curr_present:
                        count_10 += 1
                    elif not prev_present and curr_present:
                        count_01 += 1
                p_given_appeared = count_11 / max(count_11 + count_10, 1)
                p_given_absent = count_01 / max(count_01 + (t - 1 - count_11 - count_10 - count_10), 1)
                # Use last state
                last_state = presence[t - 1, ni] > 0
                markov_prob = p_given_appeared if last_state else p_given_absent
                features.append(markov_prob)

        # === 11. BAYESIAN PROBABILITY ===
        for presence, n_range in [
            (main_presence[:t], n_main),
            (sub_presence[:t], n_sub),
        ]:
            total_appearances = presence.sum()
            for ni in range(n_range):
                count = presence[:, ni].sum()
                # Beta(1,1) posterior mean
                bayes_p = (count + 1) / (total_appearances + n_range)
                features.append(bayes_p)

        # === 12. TIME-SERIES TREND (exponential smoothing) ===
        alpha = 0.3
        for presence, n_range in [
            (main_presence[:t], n_main),
            (sub_presence[:t], n_sub),
        ]:
            for ni in range(n_range):
                smoothed = 0.5
                for tt in range(t):
                    smoothed = alpha * presence[tt, ni] + (1 - alpha) * smoothed
                features.append(smoothed)

        # === 13. PAIR CO-OCCURRENCE (top 10 pairs) ===
        # For the most common number pairs, check if last draw had them
        if t > 10:
            # Find top pairs from full history
            pair_counts = Counter()
            for tt in range(t):
                mn = sorted(main_nums[tt])
                for i in range(len(mn)):
                    for j in range(i + 1, len(mn)):
                        pair_counts[(mn[i], mn[j])] += 1
            top_pairs = pair_counts.most_common(10)
            last_draw_set = set(main_nums[t - 1])
            for (a, b), _ in top_pairs:
                features.append(1.0 if (a in last_draw_set and b in last_draw_set) else 0.0)
        else:
            features.extend([0.0] * 10)

        X_rows.append(features)
        Y_rows.append(all_labels[t])

    # Build feature names
    fnames = []
    fnames.extend([f"full_freq_m{n}" for n in range(n_main)])
    fnames.extend([f"full_freq_s{n}" for n in range(n_sub)])
    for ws in window_sizes:
        fnames.extend([f"win{ws}_freq_m{n}" for n in range(n_main)])
        fnames.extend([f"win{ws}_freq_s{n}" for n in range(n_sub)])
    for ws in window_sizes:
        fnames.extend([f"win{ws}_hot_m", f"win{ws}_cold_m", f"win{ws}_hot_s", f"win{ws}_cold_s"])
    for ws in window_sizes:
        fnames.extend([f"win{ws}_oe_mean_m", f"win{ws}_oe_std_m",
                       f"win{ws}_oe_mean_s", f"win{ws}_oe_std_s"])
    for ws in window_sizes:
        fnames.extend([f"win{ws}_sum_mean_m", f"win{ws}_sum_std_m",
                       f"win{ws}_sum_mean_s", f"win{ws}_sum_std_s"])
    for ws in window_sizes:
        fnames.extend([f"win{ws}_span_mean", f"win{ws}_span_std"])
    for ws in window_sizes:
        fnames.extend([f"win{ws}_ac_mean", f"win{ws}_ac_std"])
    for ws in window_sizes:
        fnames.extend([f"win{ws}_consec_mean", f"win{ws}_consec_std"])
    fnames.extend([f"poisson_m{n}" for n in range(n_main)])
    fnames.extend([f"poisson_s{n}" for n in range(n_sub)])
    fnames.extend([f"markov_m{n}" for n in range(n_main)])
    fnames.extend([f"markov_s{n}" for n in range(n_sub)])
    fnames.extend([f"bayes_m{n}" for n in range(n_main)])
    fnames.extend([f"bayes_s{n}" for n in range(n_sub)])
    fnames.extend([f"ts_m{n}" for n in range(n_main)])
    fnames.extend([f"ts_s{n}" for n in range(n_sub)])
    for pi in range(10):
        fnames.append(f"toppair_{pi}")

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(Y_rows, dtype=np.float64)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    return X, y, fnames


def _ac_value(nums: np.ndarray) -> int:
    """AC value (complexity coefficient)."""
    n = len(nums)
    if n <= 1:
        return 0
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(abs(int(nums[i]) - int(nums[j])))
    return len(diffs) - (n - 1)


def _count_consecutive(nums: np.ndarray) -> int:
    """Count consecutive pairs."""
    s = sorted(nums)
    count = 0
    for i in range(len(s) - 1):
        if s[i + 1] - s[i] == 1:
            count += 1
    return count


def compute_feature_importance(model, feature_names: List[str], top_k: int = 20):
    """
    Extract feature importance from a fitted sklearn/XGBoost/LightGBM model.
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return {}

        indices = np.argsort(importances)[::-1][:top_k]
        return {
            feature_names[i]: float(importances[i])
            for i in indices
            if i < len(feature_names)
        }
    except Exception:
        return {}
