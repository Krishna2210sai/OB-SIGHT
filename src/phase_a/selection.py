"""Fold-specific feature selectors and stability analysis.

Every selector is fitted on the training fold only.  The protected clinical
features are forced into every subset so that clinically important variables
are not removed solely because their univariate association is weak.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from src.phase_a.config import (
    CORRELATION_THRESHOLD,
    L1_C,
    MI_K,
    PROTECTED_FEATURES,
    RF_K,
    RFE_K,
)


class BaseSelector:
    """Small sklearn-compatible selector with explicit feature names."""

    method = "base"

    def __init__(self, feature_names: Sequence[str], protected: Sequence[str] = PROTECTED_FEATURES, seed: int = 42):
        self.feature_names = list(feature_names)
        self.protected = [name for name in protected if name in self.feature_names]
        self.seed = seed
        self.support_: Optional[np.ndarray] = None
        self.importances_: Optional[np.ndarray] = None
        self.selected_names_: List[str] = []

    def _finalise(self, support: np.ndarray, importances: Optional[np.ndarray] = None) -> "BaseSelector":
        support = np.asarray(support, dtype=bool).copy()
        for name in self.protected:
            support[self.feature_names.index(name)] = True
        self.support_ = support
        self.selected_names_ = [name for name, keep in zip(self.feature_names, support) if keep]
        self.importances_ = None if importances is None else np.asarray(importances, dtype=float)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSelector":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.support_ is None:
            raise RuntimeError("Selector is not fitted")
        return np.asarray(X)[:, self.support_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        if self.support_ is None:
            raise RuntimeError("Selector is not fitted")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def selected_features(self) -> List[str]:
        return list(self.selected_names_)


class NoSelection(BaseSelector):
    method = "none"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NoSelection":
        return self._finalise(np.ones(len(self.feature_names), dtype=bool))


class CorrelationFilter(BaseSelector):
    """Remove one feature from every pair with |Pearson r| >= threshold."""

    method = "correlation"

    def __init__(self, feature_names, protected=PROTECTED_FEATURES, seed: int = 42, threshold: float = CORRELATION_THRESHOLD):
        super().__init__(feature_names, protected, seed)
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CorrelationFilter":
        X = np.asarray(X, dtype=float)
        corr = pd.DataFrame(X, columns=self.feature_names).corr().abs()
        keep = np.ones(len(self.feature_names), dtype=bool)
        protected_set = set(self.protected)
        for i, a in enumerate(self.feature_names):
            if not keep[i]:
                continue
            for j in range(i + 1, len(self.feature_names)):
                if not keep[j]:
                    continue
                value = corr.iloc[i, j]
                if not np.isfinite(value) or value < self.threshold:
                    continue
                if a in protected_set and self.feature_names[j] not in protected_set:
                    keep[j] = False
                elif self.feature_names[j] in protected_set and a not in protected_set:
                    keep[i] = False
                    break
                else:
                    keep[j] = False
        return self._finalise(keep)


class MutualInformationSelector(BaseSelector):
    method = "mutual_info"

    def __init__(self, feature_names, protected=PROTECTED_FEATURES, seed: int = 42, k: int = MI_K):
        super().__init__(feature_names, protected, seed)
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MutualInformationSelector":
        X = np.asarray(X, dtype=float)
        mi = mutual_info_classif(X, y, random_state=self.seed)
        order = np.argsort(mi)[::-1]
        support = np.zeros(len(self.feature_names), dtype=bool)
        support[order[: self.k]] = True
        return self._finalise(support, mi)


class RFESelector(BaseSelector):
    method = "rfe"

    def __init__(self, feature_names, protected=PROTECTED_FEATURES, seed: int = 42, k: int = RFE_K):
        super().__init__(feature_names, protected, seed)
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFESelector":
        X = np.asarray(X, dtype=float)
        estimator = LogisticRegression(
            class_weight="balanced", C=0.5, max_iter=3000, random_state=self.seed
        )
        rfe = RFE(estimator, n_features_to_select=self.k, step=0.1)
        rfe.fit(X, y)
        return self._finalise(rfe.support_, rfe.ranking_)


class L1Selector(BaseSelector):
    method = "l1"

    def __init__(self, feature_names, protected=PROTECTED_FEATURES, seed: int = 42, C: float = L1_C, max_features: int = 30):
        super().__init__(feature_names, protected, seed)
        self.C = C
        self.max_features = max_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "L1Selector":
        X = np.asarray(X, dtype=float)
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=self.C,
            class_weight="balanced",
            max_iter=5000,
            random_state=self.seed,
        )
        model.fit(X, y)
        coef = np.abs(model.coef_[0])
        support = coef > 1e-8
        protected_set = set(self.protected)
        if support.sum() > self.max_features:
            order = np.argsort(coef)[::-1]
            keep = set(order[: self.max_features].tolist())
            keep |= {self.feature_names.index(name) for name in protected_set}
            support = np.zeros(len(self.feature_names), dtype=bool)
            support[sorted(keep)] = True
        if support.sum() < 5:
            order = np.argsort(coef)[::-1]
            support = np.zeros(len(self.feature_names), dtype=bool)
            support[order[: min(10, len(order))]] = True
        return self._finalise(support, coef)


class RandomForestImportanceSelector(BaseSelector):
    method = "rf_importance"

    def __init__(self, feature_names, protected=PROTECTED_FEATURES, seed: int = 42, k: int = RF_K, n_estimators: int = 400):
        super().__init__(feature_names, protected, seed)
        self.k = k
        self.n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestImportanceSelector":
        X = np.asarray(X, dtype=float)
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced_subsample",
            random_state=self.seed,
            n_jobs=-1,
            min_samples_leaf=3,
        )
        model.fit(X, y)
        importance = model.feature_importances_
        order = np.argsort(importance)[::-1]
        support = np.zeros(len(self.feature_names), dtype=bool)
        support[order[: self.k]] = True
        return self._finalise(support, importance)


SELECTOR_REGISTRY = {
    "none": NoSelection,
    "correlation": CorrelationFilter,
    "mutual_info": MutualInformationSelector,
    "rfe": RFESelector,
    "l1": L1Selector,
    "rf_importance": RandomForestImportanceSelector,
}

SELECTOR_LABELS = {
    "none": "No selection",
    "correlation": "Correlation filter (|r| >= 0.90)",
    "mutual_info": "Mutual information (top 15 + protected)",
    "rfe": "RFE logistic regression (15 + protected)",
    "l1": "L1 logistic regression + protected",
    "rf_importance": "Random forest importance (15 + protected)",
}


def make_selector(method: str, feature_names: Sequence[str], seed: int = 42) -> BaseSelector:
    if method not in SELECTOR_REGISTRY:
        raise ValueError(f"Unknown selector: {method}")
    return SELECTOR_REGISTRY[method](feature_names=feature_names, seed=seed)


def selection_stability(selections: Dict[str, Iterable[Sequence[str]]], all_features: Sequence[str]) -> Tuple[pd.DataFrame, Dict]:
    """Selection frequency and pairwise Jaccard stability across folds."""
    rows = []
    method_summary: Dict[str, Dict] = {}
    for method, fold_sets in selections.items():
        sets = [set(items) for items in fold_sets]
        n = len(sets)
        counts = {name: sum(name in items for items in sets) for name in all_features}
        jaccards = []
        for a, b in combinations(range(n), 2):
            union = sets[a] | sets[b]
            jaccards.append(len(sets[a] & sets[b]) / len(union) if union else 1.0)
        union_all = set().union(*sets) if sets else set()
        intersection_all = set.intersection(*sets) if sets else set()
        method_summary[method] = {
            "n_folds": n,
            "mean_selected_count": float(np.mean([len(items) for items in sets])) if sets else 0.0,
            "union_count": len(union_all),
            "intersection_count": len(intersection_all),
            "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
            "features_selected_in_all_folds": sorted(intersection_all),
            "union_features": sorted(union_all),
        }
        for name in all_features:
            rows.append(
                {
                    "method": method,
                    "feature": name,
                    "selected_folds": counts[name],
                    "n_folds": n,
                    "selection_frequency": counts[name] / n if n else 0.0,
                    "ever_selected": int(counts[name] > 0),
                    "selected_in_all_folds": int(counts[name] == n and n > 0),
                }
            )
    return pd.DataFrame(rows), method_summary
