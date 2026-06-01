"""
Machine Learning prediction layer for lottery forecasting.

Trains ensemble of tree-based models (RandomForest, XGBoost, LightGBM, CatBoost)
on rich feature vectors to predict per-number probability distributions.

Features:
- Automatic feature engineering
- Feature normalization
- Feature importance analysis
- Auto model training with fallback chain
- Per-number prediction probability
- Combo ML score
- Model confidence
- Feature contribution
"""
import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from utils.helpers import get_logger


class MLPredictor:
    """
    Machine Learning predictor for lottery numbers.

    Trains multiple tree-based models and ensembles their predictions.
    Gracefully degrades when optional packages are missing.
    """

    def __init__(self, cfg, random_state: int = 42):
        self.cfg = cfg
        self.random_state = random_state
        self.logger = get_logger(cfg)
        self._models = {}
        self._feature_names = []
        self._is_fitted = False
        self._feature_importance = {}
        self._model_scores = {}  # model_name -> validation score

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Train all available ML models on feature matrix X.

        Uses sklearn RandomForest as base, plus XGBoost/LightGBM/CatBoost if available.
        Each model predicts each number position independently.
        """
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        n_samples, n_features = X.shape
        total_count = y.shape[1]

        if n_samples < 10:
            self.logger.warning("Too few samples (%d) for ML, skipping", n_samples)
            self._is_fitted = False
            return

        self.logger.info("Training ML models on %d samples with %d features", n_samples, n_features)

        # ---- RandomForest (always available) ----
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=self.random_state,
            n_jobs=-1, verbose=0,
        )
        rf.fit(X, y)
        self._models["random_forest"] = rf
        self._model_scores["random_forest"] = self._cv_score(rf, X, y)

        # ---- XGBoost (if available) ----
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, verbosity=0,
                objective="reg:squarederror",
            )
            xgb_model.fit(X, y)
            self._models["xgboost"] = xgb_model
            self._model_scores["xgboost"] = self._cv_score(xgb_model, X, y)
            self.logger.info("XGBoost trained successfully")
        except (ImportError, Exception) as e:
            self.logger.info("XGBoost not available: %s", e)

        # ---- LightGBM (if available) ----
        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, verbose=-1,
            )
            lgb_model.fit(X, y)
            self._models["lightgbm"] = lgb_model
            self._model_scores["lightgbm"] = self._cv_score(lgb_model, X, y)
            self.logger.info("LightGBM trained successfully")
        except (ImportError, Exception) as e:
            self.logger.info("LightGBM not available: %s", e)

        # ---- CatBoost (if available) ----
        try:
            import catboost as cb
            cb_model = cb.CatBoostRegressor(
                iterations=200, depth=6, learning_rate=0.1,
                random_seed=self.random_state, verbose=False,
            )
            cb_model.fit(X, y)
            self._models["catboost"] = cb_model
            self._model_scores["catboost"] = self._cv_score(cb_model, X, y)
            self.logger.info("CatBoost trained successfully")
        except (ImportError, Exception) as e:
            self.logger.info("CatBoost not available: %s", e)

        # Feature importance
        self._compute_feature_importance()

        self._is_fitted = True
        self.logger.info(
            "ML training complete: %d models, best=%.4f",
            len(self._models),
            max(self._model_scores.values()) if self._model_scores else 0,
        )

    def _cv_score(self, model, X: np.ndarray, y: np.ndarray, folds: int = 3) -> float:
        """Quick cross-validation score (negative MSE -> higher is better)."""
        from sklearn.model_selection import cross_val_score
        try:
            scores = cross_val_score(model, X, y, cv=min(folds, len(X) // 3),
                                     scoring="neg_mean_squared_error", n_jobs=1)
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def _compute_feature_importance(self):
        """Aggregate feature importance across all models."""
        importance = {}
        for mname, model in self._models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
                else:
                    continue
                for i, v in enumerate(imp):
                    if i < len(self._feature_names):
                        importance[self._feature_names[i]] = \
                            importance.get(self._feature_names[i], 0) + v / len(self._models)
            except Exception:
                continue

        # Normalize
        total = sum(importance.values())
        if total > 0:
            self._feature_importance = {
                k: round(v / total * 100, 2)
                for k, v in sorted(importance.items(), key=lambda x: -x[1])
            }

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict number probabilities using ensemble of trained models.

        Parameters
        ----------
        X : feature matrix, shape (1, n_features)

        Returns
        -------
        dict with:
            main_probs : per-number probabilities for main
            sub_probs : per-number probabilities for sub
            ml_confidence : model confidence score
            model_contributions : per-model contribution
        """
        if not self._is_fitted or not self._models:
            return self._fallback_uniform()

        n_main = self.cfg.main_max - self.cfg.main_min + 1
        n_sub = self.cfg.sub_max - self.cfg.sub_min + 1
        total_count = n_main + n_sub

        # Get predictions from each model, weighted by CV score
        weighted_pred = np.zeros(total_count)
        total_weight = 0
        contributions = {}

        for mname, model in self._models.items():
            try:
                raw_pred = model.predict(X)[0]
            except Exception:
                raw_pred = np.ones(total_count) * 0.5

            score = self._model_scores.get(mname, 1.0)
            # Convert CV score (negative MSE) to weight
            weight = np.exp(score) if score > -100 else 1.0
            weighted_pred += raw_pred * weight
            contributions[mname] = {
                "weight": round(weight, 2),
                "prediction": raw_pred.tolist(),
            }
            total_weight += weight

        if total_weight > 0:
            weighted_pred /= total_weight

        # Split into main and sub probs
        main_raw = weighted_pred[:n_main]
        sub_raw = weighted_pred[n_main:]

        # Clip to valid range, then normalize to probabilities
        main_clipped = np.maximum(main_raw, self.cfg.main_min)
        main_clipped = np.minimum(main_clipped, self.cfg.main_max)
        # Convert to probability: closer to predicted = higher prob
        main_probs = np.zeros(n_main)
        for i in range(n_main):
            num = i + self.cfg.main_min
            # Probability based on how close the prediction is to this number
            diff = abs(main_raw[i] - num)
            main_probs[i] = np.exp(-diff * 0.5)
        main_probs = main_probs / main_probs.sum()

        sub_probs = np.ones(n_sub) / n_sub
        if n_sub > 0:
            for i in range(n_sub):
                num = i + self.cfg.sub_min
                diff = abs(sub_raw[i] - num) if i < len(sub_raw) else 0
                sub_probs[i] = np.exp(-diff * 0.5)
            sub_probs = sub_probs / sub_probs.sum()

        # Confidence: variance of predictions across models
        preds = []
        for mname, model in self._models.items():
            try:
                preds.append(model.predict(X)[0])
            except Exception:
                pass
        if len(preds) > 1:
            pred_std = np.std(preds, axis=0).mean()
            confidence = np.exp(-pred_std)  # lower std = higher confidence
        else:
            confidence = 0.5

        return {
            "main_probs": main_probs,
            "sub_probs": sub_probs,
            "ml_confidence": round(float(confidence), 4),
            "model_contributions": contributions,
            "num_models": len(self._models),
        }

    def _fallback_uniform(self) -> Dict:
        """Uniform distribution fallback."""
        n_main = self.cfg.main_max - self.cfg.main_min + 1
        n_sub = self.cfg.sub_max - self.cfg.sub_min + 1
        return {
            "main_probs": np.ones(n_main) / n_main,
            "sub_probs": np.ones(n_sub) / n_sub,
            "ml_confidence": 0.0,
            "model_contributions": {},
            "num_models": 0,
        }

    def get_feature_importance(self, top_k: int = 20) -> Dict[str, float]:
        """Return top-K feature importance."""
        sorted_imp = sorted(self._feature_importance.items(), key=lambda x: -x[1])
        return dict(sorted_imp[:top_k])

    def score_combination(self, main: List[int], sub: List[int],
                          X: np.ndarray = None) -> float:
        """
        Score a single number combination using ML models.

        Returns a score between 0 and 100.
        """
        if not self._is_fitted:
            return 50.0  # neutral

        # If no feature vector provided, use uniform
        if X is None:
            probs = self._fallback_uniform()
        else:
            probs = self.predict_proba(X)

        main_probs = probs["main_probs"]
        sub_probs = probs["sub_probs"]

        # Score based on how well the combination matches predicted distribution
        main_score = sum(main_probs[n - self.cfg.main_min] for n in main)
        sub_score = sum(sub_probs[n - self.cfg.sub_min] for n in sub)

        expected_main = self.cfg.main_count / len(main_probs)
        expected_sub = self.cfg.sub_count / max(len(sub_probs), 1)

        # Ratio to expected: 1.0 = average, >1 = better
        main_ratio = main_score / max(expected_main, 0.001)
        sub_ratio = sub_score / max(expected_sub, 0.001)

        combined = (main_ratio * self.cfg.main_count + sub_ratio * self.cfg.sub_count)
        combined /= (self.cfg.main_count + self.cfg.sub_count)

        # Scale to 0-100
        ml_score = min(100, combined * 50)
        return round(ml_score, 2)
