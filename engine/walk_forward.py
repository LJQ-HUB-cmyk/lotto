"""
Walk-Forward backtesting engine for lottery prediction models.

Implements complete rolling backtest:
- Uses first N draws as training, predicts next draw
- Rolls forward one draw at a time, adding real results to training
- Tracks per-model performance metrics
- Records single-number hit rates, combo overlap, top-N hit rates,
  sum/odd-even/span/AC/consecutive hit rates, hot/cold hit rates
- Generates model performance database for adaptive weighting
"""
import json
import math
import time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from utils.helpers import get_logger


class WalkForwardBacktester:
    """
    Complete walk-forward backtesting engine.

    For each test period:
      1. Train all models on data[:train_end]
      2. Generate predictions for period train_end
      3. Compare with actual draw at train_end
      4. Record per-model and ensemble metrics
      5. Roll forward: train_end += 1
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg,
        initial_train: int = 100,
        test_window: int = 50,
        step: int = 1,
        num_groups: int = 5,
        random_state: int = 42,
    ):
        self.df = df.sort_values("period", ascending=True).reset_index(drop=True)
        self.cfg = cfg
        self.initial_train = max(initial_train, 50)
        self.test_window = test_window
        self.step = step
        self.num_groups = num_groups
        self.random_state = random_state
        self.logger = get_logger(cfg)

        self.total = len(self.df)

        # Performance database
        self.performance_db = {
            "by_model": defaultdict(list),   # model_name -> list of {period, hits, ...}
            "by_period": [],                  # list of {period, model_hits, ensemble_hits, ...}
            "by_window": defaultdict(list),   # window_size -> list of metrics
            "single_number_hits": defaultdict(list),  # number -> [hit_count, total_predicted]
            "hit_distribution": defaultdict(int),     # total_hits -> count
        }

    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """Run the full walk-forward backtest."""
        self.logger.info(
            "Starting walk-forward backtest: train=%d, test=%d, step=%d, total=%d",
            self.initial_train, self.test_window, self.step, self.total
        )

        if self.total < self.initial_train + 10:
            return {"error": f"数据不足: 需要至少 {self.initial_train + 10} 期, 当前 {self.total} 期"}

        from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
        from models.timeseries import ExponentialSmoothingModel

        test_end = min(self.initial_train + self.test_window, self.total)
        results = []
        model_names = ["frequency", "poisson", "monte_carlo", "exponential_smoothing",
                       "bayesian", "markov_chain", "sliding_window", "ensemble"]

        start_time = time.time()

        for train_end in range(self.initial_train, test_end, self.step):
            train_df = self.df.iloc[:train_end]
            actual_row = self.df.iloc[train_end]
            actual_main = sorted([int(actual_row[c]) for c in self.cfg.main_cols])
            actual_sub = sorted([int(actual_row[c]) for c in self.cfg.sub_cols])
            actual_set_main = set(actual_main)
            actual_set_sub = set(actual_sub)
            period = actual_row.get("period", train_end)

            main_nums = np.array([
                sorted([int(r[c]) for c in self.cfg.main_cols])
                for _, r in train_df.iterrows()
            ])
            sub_nums = np.array([
                sorted([int(r[c]) for c in self.cfg.sub_cols])
                for _, r in train_df.iterrows()
            ])

            n_draws = len(main_nums)
            n_main = self.cfg.main_max - self.cfg.main_min + 1
            n_sub = self.cfg.sub_max - self.cfg.sub_min + 1

            # ---- Compute all model scores ----
            model_scores = {}  # model_name -> main_probs[], sub_probs[]

            # 1. Full frequency
            main_counts_full = np.zeros(n_main)
            sub_counts_full = np.zeros(n_sub)
            for row in main_nums:
                for n in row:
                    main_counts_full[n - self.cfg.main_min] += 1
            for row in sub_nums:
                for n in row:
                    sub_counts_full[n - self.cfg.sub_min] += 1
            mf = (main_counts_full + 1) / (main_counts_full.sum() + n_main)
            sf = (sub_counts_full + 1) / (sub_counts_full.sum() + n_sub)
            model_scores["frequency"] = (mf, sf)

            # 2. Sliding window (last 50)
            w = min(50, n_draws)
            mwc = np.zeros(n_main)
            swc = np.zeros(n_sub)
            for row in main_nums[:w]:
                for n in row:
                    mwc[n - self.cfg.main_min] += 1
            for row in sub_nums[:w]:
                for n in row:
                    swc[n - self.cfg.sub_min] += 1
            mw = (mwc + 1) / (mwc.sum() + n_main)
            sw = (swc + 1) / (swc.sum() + n_sub)
            model_scores["sliding_window"] = (mw, sw)

            # 3. Bayesian
            total_main = main_counts_full.sum()
            total_sub = sub_counts_full.sum()
            mb = (main_counts_full + 1) / (total_main + n_main)
            sb = (sub_counts_full + 1) / (total_sub + n_sub)
            model_scores["bayesian"] = (mb, sb)

            # 4. Poisson overdue
            mp = np.ones(n_main) * 0.5
            sp = np.ones(n_sub) * 0.5
            for i in range(n_main):
                num = i + self.cfg.main_min
                appearances = [idx for idx, row in enumerate(main_nums) if num in row]
                if appearances:
                    last_seen = appearances[-1]
                    gap = n_draws - 1 - last_seen
                    lam = max(main_counts_full[i] / n_draws * n_draws, 0.5)
                    surv = math.exp(-gap / lam) if lam > 0 else 0
                    mp[i] = 1.0 - surv
                else:
                    mp[i] = 1.0
            for i in range(n_sub):
                num = i + self.cfg.sub_min
                appearances = [idx for idx, row in enumerate(sub_nums) if num in row]
                if appearances:
                    last_seen = appearances[-1]
                    gap = n_draws - 1 - last_seen
                    lam = max(sub_counts_full[i] / n_draws * n_draws, 0.5)
                    surv = math.exp(-gap / lam) if lam > 0 else 0
                    sp[i] = 1.0 - surv
                else:
                    sp[i] = 1.0
            mp = mp / mp.sum()
            sp = sp / sp.sum()
            model_scores["poisson"] = (mp, sp)

            # 5. Markov chain
            mm = np.ones(n_main) * 0.5
            sm = np.ones(n_sub) * 0.5
            for i, n in enumerate(range(self.cfg.main_min, self.cfg.main_max + 1)):
                count_11 = count_10 = count_01 = 0
                for idx in range(1, n_draws):
                    prev = n in main_nums[idx - 1]
                    curr = n in main_nums[idx]
                    if prev and curr: count_11 += 1
                    elif prev and not curr: count_10 += 1
                    elif not prev and curr: count_01 += 1
                p_ga = count_11 / max(count_11 + count_10, 1)
                p_ab = count_01 / max(count_01 + (n_draws - count_11 - count_10 - count_10), 1)
                mm[i] = p_ga if (n in main_nums[0]) else p_ab
            for i, n in enumerate(range(self.cfg.sub_min, self.cfg.sub_max + 1)):
                count_11 = count_10 = count_01 = 0
                for idx in range(1, n_draws):
                    prev = n in sub_nums[idx - 1]
                    curr = n in sub_nums[idx]
                    if prev and curr: count_11 += 1
                    elif prev and not curr: count_10 += 1
                    elif not prev and curr: count_01 += 1
                p_ga = count_11 / max(count_11 + count_10, 1)
                p_ab = count_01 / max(count_01 + (n_draws - count_11 - count_10 - count_10), 1)
                sm[i] = p_ga if (n in sub_nums[0]) else p_ab
            mm = mm / mm.sum()
            sm = sm / sm.sum()
            model_scores["markov_chain"] = (mm, sm)

            # 6. Exponential smoothing
            mt = np.ones(n_main) * 0.5
            st_s = np.ones(n_sub) * 0.5
            alpha = 0.3
            for i in range(n_main):
                s = 0.5
                for idx in range(n_draws):
                    appeared = 1.0 if (i + self.cfg.main_min) in main_nums[idx] else 0.0
                    s = alpha * appeared + (1 - alpha) * s
                mt[i] = s
            for i in range(n_sub):
                s = 0.5
                for idx in range(n_draws):
                    appeared = 1.0 if (i + self.cfg.sub_min) in sub_nums[idx] else 0.0
                    s = alpha * appeared + (1 - alpha) * s
                st_s[i] = s
            mt = mt / mt.sum()
            st_s = st_s / st_s.sum()
            model_scores["exponential_smoothing"] = (mt, st_s)

            # 7. Ensemble (equal weights first, will use adaptive later)
            base_weights = {
                "frequency": 0.20, "sliding_window": 0.15, "bayesian": 0.10,
                "poisson": 0.15, "markov_chain": 0.15, "exponential_smoothing": 0.15,
                "monte_carlo": 0.10,
            }
            # Adjust with adaptive weights from performance DB
            adaptive_weights = self._get_adaptive_weights(base_weights)
            men = np.zeros(n_main)
            sen = np.zeros(n_sub)
            for mname, (mp_, sp_) in model_scores.items():
                w = adaptive_weights.get(mname, base_weights.get(mname, 0.10))
                men += mp_ * w
                sen += sp_ * w
            men = men / men.sum()
            sen = sen / sen.sum()
            model_scores["ensemble"] = (men, sen)

            # Extract weighted candidates and evaluate all models
            main_range = list(range(self.cfg.main_min, self.cfg.main_max + 1))
            sub_range = list(range(self.cfg.sub_min, self.cfg.sub_max + 1))

            period_entry = {
                "period": str(period),
                "train_size": n_draws,
                "actual_main": actual_main,
                "actual_sub": actual_sub,
                "model_results": {},
            }

            for mname, (main_probs, sub_probs) in model_scores.items():
                # Pick top N main numbers by probability
                main_ranked = np.argsort(main_probs)[::-1]
                sub_ranked = np.argsort(sub_probs)[::-1]
                top_main = sorted([main_range[i] for i in main_ranked[:self.cfg.main_count]])
                top_sub = sorted([sub_range[i] for i in sub_ranked[:self.cfg.sub_count]])

                main_hits = len(set(top_main) & actual_set_main)
                sub_hits = len(set(top_sub) & actual_set_sub)
                total_hits = main_hits + sub_hits

                # Top-N hit rate (for each N)
                topn_hits = {}
                for topn in [5, 10, 20, 30]:
                    tm = set(main_range[i] for i in main_ranked[:topn])
                    topn_hits[f"top{topn}"] = len(tm & actual_set_main)

                # Structural metrics
                pred_sum = sum(top_main)
                actual_sum = sum(actual_main)
                sum_hit = 1 if abs(pred_sum - actual_sum) <= 15 else 0

                pred_odds = sum(1 for n in top_main if n % 2 == 1)
                actual_odds = sum(1 for n in actual_main if n % 2 == 1)
                oe_hit = 1 if abs(pred_odds - actual_odds) <= 1 else 0

                pred_span = max(top_main) - min(top_main)
                actual_span = max(actual_main) - min(actual_main)
                span_hit = 1 if abs(pred_span - actual_span) <= 5 else 0

                pred_ac = _ac_value(top_main)
                actual_ac = _ac_value(actual_main)
                ac_hit = 1 if abs(pred_ac - actual_ac) <= 2 else 0

                pred_consec = _count_consecutive(top_main)
                actual_consec = _count_consecutive(actual_main)
                consec_hit = 1 if pred_consec == actual_consec else 0

                # Hot/cold hit rate
                hot_main = set(main_range[i] for i in main_ranked[:int(n_main * 0.2)])
                cold_main = set(main_range[i] for i in main_ranked[-int(n_main * 0.2):])
                hot_hits = len(hot_main & actual_set_main)
                cold_hits = len(cold_main & actual_set_main)

                m_result = {
                    "main_hits": main_hits,
                    "sub_hits": sub_hits,
                    "total_hits": total_hits,
                    "top5_hits": topn_hits.get("top5", 0),
                    "top10_hits": topn_hits.get("top10", 0),
                    "top20_hits": topn_hits.get("top20", 0),
                    "top30_hits": topn_hits.get("top30", 0),
                    "sum_hit": sum_hit,
                    "oe_hit": oe_hit,
                    "span_hit": span_hit,
                    "ac_hit": ac_hit,
                    "consec_hit": consec_hit,
                    "hot_hits": hot_hits,
                    "cold_hits": cold_hits,
                    "pred_main": top_main,
                    "pred_sub": top_sub,
                }
                period_entry["model_results"][mname] = m_result

                # Update performance DB
                self.performance_db["by_model"][mname].append(m_result)
                # Single number hits
                for n in top_main:
                    self.performance_db["single_number_hits"][n].append(
                        1 if n in actual_set_main else 0
                    )
                self.performance_db["hit_distribution"][total_hits] += 1

            self.performance_db["by_period"].append(period_entry)
            results.append(period_entry)

            if verbose and (train_end % 10 == 0 or train_end == test_end - 1):
                ens = period_entry["model_results"].get("ensemble", {})
                ens_hits = ens.get("total_hits", 0)
                self.logger.info(
                    "  Period %s (train=%d): ensemble hits=%d, best=%s",
                    period, n_draws, ens_hits,
                    max(
                        (r["total_hits"], nm)
                        for nm, r in period_entry["model_results"].items()
                    )
                )

        elapsed = time.time() - start_time
        self.logger.info(
            "Backtest complete: %d periods in %.1fs (%.1f ms/period)",
            len(results), elapsed, elapsed / max(len(results), 1) * 1000
        )

        return self._compile_stats(results)

    def _get_adaptive_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Compute adaptive weights based on recent model performance."""
        if not self.performance_db["by_period"]:
            return base_weights

        # Use last 20 periods for adaptive weights
        recent_window = min(20, len(self.performance_db["by_period"]))
        recent = self.performance_db["by_period"][-recent_window:]

        # Compute mean total_hits per model with exponential decay
        weights = {}
        for mname in base_weights:
            hits = []
            for i, entry in enumerate(recent):
                mr = entry["model_results"].get(mname)
                if mr:
                    # Exponential decay: newer periods weighted more
                    decay = 0.9 ** (recent_window - i - 1)
                    hits.append(mr["total_hits"] * decay)
            if hits:
                weights[mname] = sum(hits) / max(len(hits), 1)
            else:
                weights[mname] = 0

        # Convert to softmax probabilities
        if sum(weights.values()) > 0:
            exp_w = {k: math.exp(v * 2) for k, v in weights.items()}
            total = sum(exp_w.values())
            weights = {k: v / total for k, v in exp_w.items()}
        else:
            weights = dict(base_weights)

        # Diversity constraint: no single model > 35%
        max_w = 0.35
        for k in weights:
            weights[k] = min(weights[k], max_w)
        # Renormalize
        wsum = sum(weights.values())
        if wsum > 0:
            weights = {k: v / wsum for k, v in weights.items()}

        return weights

    def _compile_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Compile comprehensive statistics from results."""
        if not results:
            return {"error": "No backtest results"}

        stats = {
            "total_tests": len(results),
            "periods": [r["period"] for r in results],
            "by_model": {},
            "ensemble": {},
            "hit_distribution": dict(self.performance_db["hit_distribution"]),
            "summary": {},
        }

        for mname in ["ensemble", "frequency", "poisson", "monte_carlo",
                       "exponential_smoothing", "bayesian", "markov_chain",
                       "sliding_window"]:
            hits = []
            main_hits = []
            sub_hits = []
            top5_list = []
            top10_list = []
            top20_list = []
            sum_hits = []
            oe_hits = []
            span_hits = []
            ac_hits = []
            consec_hits = []
            hot_hits_list = []
            cold_hits_list = []

            for entry in results:
                mr = entry["model_results"].get(mname)
                if mr:
                    hits.append(mr["total_hits"])
                    main_hits.append(mr["main_hits"])
                    sub_hits.append(mr["sub_hits"])
                    top5_list.append(mr["top5_hits"])
                    top10_list.append(mr["top10_hits"])
                    top20_list.append(mr["top20_hits"])
                    sum_hits.append(mr["sum_hit"])
                    oe_hits.append(mr["oe_hit"])
                    span_hits.append(mr["span_hit"])
                    ac_hits.append(mr["ac_hit"])
                    consec_hits.append(mr["consec_hit"])
                    hot_hits_list.append(mr["hot_hits"])
                    cold_hits_list.append(mr["cold_hits"])

            if hits:
                hit_arr = np.array(hits)
                stats["by_model"][mname] = {
                    "mean_hits": float(np.mean(hit_arr)),
                    "median_hits": float(np.median(hit_arr)),
                    "max_hits": int(np.max(hit_arr)),
                    "min_hits": int(np.min(hit_arr)),
                    "std_hits": float(np.std(hit_arr)),
                    "main_mean": float(np.mean(main_hits)),
                    "sub_mean": float(np.mean(sub_hits)),
                    "top5_mean": float(np.mean(top5_list)),
                    "top10_mean": float(np.mean(top10_list)),
                    "top20_mean": float(np.mean(top20_list)),
                    "sum_accuracy": float(np.mean(sum_hits)) * 100,
                    "oe_accuracy": float(np.mean(oe_hits)) * 100,
                    "span_accuracy": float(np.mean(span_hits)) * 100,
                    "ac_accuracy": float(np.mean(ac_hits)) * 100,
                    "consec_accuracy": float(np.mean(consec_hits)) * 100,
                    "hot_hits_mean": float(np.mean(hot_hits_list)),
                    "cold_hits_mean": float(np.mean(cold_hits_list)),
                    "hit_rate_3plus": float(np.mean(hit_arr >= 3)) * 100,
                    "hit_rate_2plus": float(np.mean(hit_arr >= 2)) * 100,
                }

        # Short-term (last 20) vs long-term (all) comparison
        for period_label, period_slice in [("short_term_20", results[-20:]), ("all", results)]:
            if len(period_slice) < 3:
                continue
            comp = {}
            for mname in self.performance_db["by_model"]:
                hits = [
                    r["model_results"][mname]["total_hits"]
                    for r in period_slice if mname in r["model_results"]
                ]
                if hits:
                    comp[mname] = float(np.mean(hits))
            if comp:
                best_model = max(comp, key=comp.get)
                stats[f"best_model_{period_label}"] = {
                    "model": best_model,
                    "mean_hits": comp[best_model],
                    "all_means": comp,
                }

        stats["summary"] = {
            "total_tests": len(results),
            "ensemble_mean": stats["by_model"].get("ensemble", {}).get("mean_hits", 0),
            "ensemble_max": stats["by_model"].get("ensemble", {}).get("max_hits", 0),
            "best_model_all": stats.get("best_model_all", {}).get("model", "N/A"),
            "best_model_short": stats.get("best_model_short_term_20", {}).get("model", "N/A"),
        }

        return stats

    def save_performance_db(self, path: Optional[Path] = None):
        """Save performance database to JSON."""
        if path is None:
            path = Path(self.cfg.data_dir) / "backtest" / "performance_db.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        db = {
            "by_model": {
                k: v for k, v in self.performance_db["by_model"].items()
            },
            "by_period": self.performance_db["by_period"],
            "hit_distribution": dict(self.performance_db["hit_distribution"]),
            "single_number_hits": {
                str(k): v for k, v in self.performance_db["single_number_hits"].items()
            },
            "timestamp": datetime.now().isoformat(),
            "cfg": self.cfg.short,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info("Performance DB saved to %s", path)
        return path


def _ac_value(nums: list) -> int:
    n = len(nums)
    if n <= 1:
        return 0
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(abs(int(nums[i]) - int(nums[j])))
    return len(diffs) - (n - 1)


def _count_consecutive(nums: list) -> int:
    s = sorted(nums)
    count = 0
    for i in range(len(s) - 1):
        if s[i + 1] - s[i] == 1:
            count += 1
    return count
