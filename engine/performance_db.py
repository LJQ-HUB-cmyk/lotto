"""
Historical performance database for model tracking and adaptive weighting.

Records, queries, and analyzes model performance across different time windows.
Provides the data foundation for the Adaptive Ensemble Weighting System.
"""
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np

from utils.helpers import get_logger


class PerformanceDB:
    """
    Historical performance database for ensemble model adaptive weighting.

    Stores per-model, per-period performance metrics and computes
    rolling evaluations with exponential weight decay.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = get_logger(cfg)
        self.db_path = Path(cfg.data_dir) / "backtest" / "performance_db.json"

        # In-memory store
        self.by_model = defaultdict(list)   # model_name -> [{period, total_hits, main_hits, ...}]
        self.by_period = []                  # [{period, timestamp, model_results}]
        self.model_performance_cache = {}    # model_name -> {mean_hits, window_means}

        self._load()

    def _load(self):
        """Load from disk if exists."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for mname, records in data.get("by_model", {}).items():
                    self.by_model[mname] = records
                self.by_period = data.get("by_period", [])
                self.logger.info(
                    "Loaded performance DB: %d models, %d periods",
                    len(self.by_model), len(self.by_period),
                )
            except Exception as e:
                self.logger.warning("Failed to load performance DB: %s", e)

    def save(self):
        """Save to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "by_model": dict(self.by_model),
            "by_period": self.by_period,
            "timestamp": datetime.now().isoformat(),
            "cfg": self.cfg.short,
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info("Performance DB saved (%d periods)", len(self.by_period))

    def record_period(self, period: str, model_results: Dict[str, Dict]):
        """
        Record results for one test period.

        model_results: {model_name: {total_hits, main_hits, sub_hits, ...}}
        """
        entry = {
            "period": str(period),
            "timestamp": datetime.now().isoformat(),
            "model_results": {},
        }
        for mname, metrics in model_results.items():
            record = {
                "period": str(period),
                "total_hits": metrics.get("total_hits", 0),
                "main_hits": metrics.get("main_hits", 0),
                "sub_hits": metrics.get("sub_hits", 0),
                "top5_hits": metrics.get("top5_hits", 0),
                "top10_hits": metrics.get("top10_hits", 0),
                "sum_hit": metrics.get("sum_hit", 0),
                "oe_hit": metrics.get("oe_hit", 0),
                "span_hit": metrics.get("span_hit", 0),
                "ac_hit": metrics.get("ac_hit", 0),
                "consec_hit": metrics.get("consec_hit", 0),
            }
            self.by_model[mname].append(record)
            entry["model_results"][mname] = record

        self.by_period.append(entry)
        # Invalidate cache
        self.model_performance_cache = {}

    # ====================================================================
    # ROLLING PERFORMANCE EVALUATION
    # ====================================================================

    def get_model_performance(
        self,
        model_name: str,
        window: int = 20,
        decay: float = 0.9,
    ) -> Optional[Dict[str, float]]:
        """
        Get rolling performance for a model with exponential weight decay.

        Parameters
        ----------
        model_name : str
        window : number of recent periods to consider
        decay  : exponential decay factor (0-1), higher = more weight on recent

        Returns
        -------
        dict with: mean_hits, weighted_mean_hits, main_mean, sub_mean,
                   hit_rate_2plus, hit_rate_3plus, trend (positive = improving)
        """
        records = self.by_model.get(model_name, [])
        if not records:
            return None

        recent = records[-window:]

        hits = [r["total_hits"] for r in recent]
        main_hits = [r["main_hits"] for r in recent]
        sub_hits = [r["sub_hits"] for r in recent]

        # Exponential weighted mean
        weights = [decay ** (len(recent) - i - 1) for i in range(len(recent))]
        wsum = sum(weights)
        if wsum > 0:
            weighted_mean = sum(h * w for h, w in zip(hits, weights)) / wsum
        else:
            weighted_mean = float(np.mean(hits))

        # Trend: slope of last 10 periods
        if len(hits) >= 10:
            recent_hits = hits[-10:]
            x = np.arange(len(recent_hits))
            slope = np.polyfit(x, recent_hits, 1)[0]
            # Normalize: slope per 100 periods
            trend = slope * 10
        else:
            trend = 0.0

        return {
            "mean_hits": float(np.mean(hits)),
            "weighted_mean_hits": float(weighted_mean),
            "main_mean": float(np.mean(main_hits)),
            "sub_mean": float(np.mean(sub_hits)),
            "hit_rate_2plus": float(np.mean(np.array(hits) >= 2)) * 100,
            "hit_rate_3plus": float(np.mean(np.array(hits) >= 3)) * 100,
            "trend": float(trend),
            "max_hits": int(np.max(hits)) if hits else 0,
            "std_hits": float(np.std(hits)) if hits else 0,
            "samples": len(hits),
        }

    def get_all_model_performance(self, window: int = 20) -> Dict[str, float]:
        """Get weighted mean_hits for all models."""
        result = {}
        for mname in self.by_model:
            perf = self.get_model_performance(mname, window)
            if perf:
                result[mname] = perf["weighted_mean_hits"]
        return result

    # ====================================================================
    # ADAPTIVE WEIGHT COMPUTATION
    # ====================================================================

    def compute_adaptive_weights(
        self,
        base_weights: Dict[str, float],
        window: int = 20,
        temperature: float = 2.0,
        max_weight: float = 0.35,
    ) -> Dict[str, float]:
        """
        Compute adaptive ensemble weights using softmax allocation.

        Parameters
        ----------
        base_weights : default weights for each model
        window       : rolling window size for performance evaluation
        temperature  : softmax temperature (higher = more aggressive weighting)
        max_weight   : maximum weight per model to prevent over-concentration

        Returns
        -------
        dict of model_name -> weight (sums to 1.0)
        """
        perf = {}
        for mname in base_weights:
            p = self.get_model_performance(mname, window)
            if p:
                # Use weighted mean + trend bonus
                score = p["weighted_mean_hits"]
                trend_bonus = max(0, p["trend"]) * 0.5
                perf[mname] = score + trend_bonus
            else:
                perf[mname] = 0.0

        # Softmax transformation
        if max(perf.values()) > 0:
            exp_scores = {}
            for mname, s in perf.items():
                exp_scores[mname] = math.exp(s * temperature / max(perf.values()))

            total = sum(exp_scores.values())
            weights = {k: v / total for k, v in exp_scores.items()}
        else:
            weights = dict(base_weights)

        # Apply max weight constraint
        for k in list(weights.keys()):
            if weights[k] > max_weight:
                excess = weights[k] - max_weight
                weights[k] = max_weight
                # Redistribute excess to other models proportionally
                others = {m: w for m, w in weights.items() if m != k and w > 0}
                other_total = sum(others.values())
                if other_total > 0:
                    for ok in others:
                        weights[ok] += excess * others[ok] / other_total

        # Normalize to exactly 1.0
        wsum = sum(weights.values())
        if wsum > 0:
            weights = {k: v / wsum for k, v in weights.items()}

        return weights

    def compute_adaptive_scores(self, X: np.ndarray = None) -> Dict[str, float]:
        """
        Compute adaptive confidence scores for each number.
        Higher = more confident based on model agreement.

        Returns dict: number -> confidence score (0-1)
        """
        scores = {}
        for model_name, records in self.by_model.items():
            if len(records) < 5:
                continue
            recent = records[-20:]
            hits = np.array([r["total_hits"] for r in recent])
            # Consistency: low variance = high confidence
            mean_hit = np.mean(hits)
            std_hit = np.std(hits) if len(hits) > 1 else 1
            cv = std_hit / max(mean_hit, 0.1)  # coefficient of variation
            scores[model_name] = float(np.exp(-cv))  # 0-1, higher = more consistent

        return scores

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary across all models."""
        summary = {
            "total_periods": len(self.by_period),
            "models": {},
            "best_model": None,
            "best_mean_hits": 0,
            "trending": [],
            "declining": [],
        }

        for mname in self.by_model:
            short = self.get_model_performance(mname, window=20)
            long = self.get_model_performance(mname, window=100)

            if short and long:
                summary["models"][mname] = {
                    "short_mean": round(short["weighted_mean_hits"], 3),
                    "long_mean": round(long["weighted_mean_hits"], 3),
                    "trend": round(short["trend"], 3),
                    "hit_rate_3plus": round(short["hit_rate_3plus"], 1),
                }

                if short["weighted_mean_hits"] > summary["best_mean_hits"]:
                    summary["best_mean_hits"] = short["weighted_mean_hits"]
                    summary["best_model"] = mname

                if short["trend"] > 0.05:
                    summary["trending"].append(mname)
                elif short["trend"] < -0.05:
                    summary["declining"].append(mname)

        return summary
