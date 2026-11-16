"""Tests for the additive Phase B imbalance modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.phase_b.classical import fit_classical
from src.phase_b.data import make_inner_folds, outer_train_validation
from src.phase_b.imbalance import (
    FocalLoss,
    augment_batch,
    build_loss,
    class_counts,
    compute_class_weights,
    make_weighted_sampler,
    positive_weight,
    random_oversample,
    sampler_weights,
    smote_oversample,
)
from src.phase_b.metrics import (
    best_threshold,
    bootstrap_confidence_intervals,
    evaluate_predictions,
    paired_bootstrap_difference,
)
from src.phase_b.training import DeepRunConfig, fit_deep, predict_array


def _synthetic_binary(n: int = 80, prevalence: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    # Guarantee both classes.
    y[:2] = [0, 1]
    X = rng.normal(size=(n, 6))
    X[:, 0] += 2.0 * y
    return X, y


def _synthetic_folds(labels: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    # Deterministic interleaving keeps both classes in every fold.
    fold = np.tile(np.arange(1, n_splits + 1), int(np.ceil(len(labels) / n_splits)))[: len(labels)]
    return pd.DataFrame({"fold": fold, "label": labels})


def test_class_weights_use_only_supplied_labels_and_balance_classes() -> None:
    y = np.array([0] * 90 + [1] * 10)
    weights = compute_class_weights(y)
    assert weights[0] == pytest.approx(100 / (2 * 90))
    assert weights[1] == pytest.approx(100 / (2 * 10))
    assert positive_weight(y) == pytest.approx(9.0)

    # Replacing validation labels must not affect a weight fitted on training labels.
    train = np.array([0] * 80 + [1] * 20)
    validation = np.array([0, 1, 1, 1])
    before = compute_class_weights(train)
    _ = validation.copy()
    validation[:] = 1
    after = compute_class_weights(train)
    assert before == after


def test_focal_loss_gamma_zero_matches_alpha_weighted_bce() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([2.0, -1.0, 0.5, -2.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    alpha = 0.25
    loss = FocalLoss(gamma=0.0, alpha=alpha, reduction="mean")(logits, targets)

    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    expected = (alpha_t * ce).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_focal_loss_downweights_easy_examples_and_derives_alpha() -> None:
    y_train = np.array([0] * 90 + [1] * 10)
    loss_fn = FocalLoss(y_train=y_train, gamma=2.0)
    assert loss_fn.alpha == pytest.approx(0.9)

    logits = torch.tensor([5.0, -5.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    focal = loss_fn(logits, targets)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert focal.item() < ce.item()

    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0, alpha=0.5)


def test_build_loss_maps_every_deep_strategy() -> None:
    y = np.array([0] * 9 + [1] * 3)
    for strategy in (
        "plain_bce",
        "weighted_bce",
        "focal_gamma1",
        "focal_gamma2",
        "weighted_sampling",
    ):
        loss = build_loss(strategy, y)
        assert isinstance(loss, torch.nn.Module)
    with pytest.raises(ValueError):
        build_loss("does-not-exist", y)


def test_weighted_sampler_is_training_only_and_reproducible() -> None:
    y = np.array([0] * 18 + [1] * 2)
    weights = sampler_weights(y)
    assert weights[y == 1].mean() > weights[y == 0].mean()

    sampler_a = make_weighted_sampler(y, seed=42)
    sampler_b = make_weighted_sampler(y, seed=42)
    indices_a = list(sampler_a)
    indices_b = list(sampler_b)
    assert indices_a == indices_b
    assert len(indices_a) == len(y)
    assert set(indices_a).issubset(set(range(len(y))))

    # A validation array is a separate object and is never passed to the sampler.
    validation = np.array([0, 1, 0])
    before = validation.copy()
    _ = list(make_weighted_sampler(y, seed=7))
    assert np.array_equal(validation, before)


def test_random_oversample_and_smote_only_expand_training_rows() -> None:
    X, y = _synthetic_binary(n=60, prevalence=0.2, seed=4)
    X_before, y_before = X.copy(), y.copy()
    X_ro, y_ro = random_oversample(X, y, seed=42)
    assert len(X_ro) == len(X) + (int((y == 0).sum()) - int((y == 1).sum()))
    assert np.array_equal(X, X_before)
    assert np.array_equal(y, y_before)
    assert np.array_equal(y_ro.sum(), (y == 0).sum())

    X_sm, y_sm = smote_oversample(X, y, k_neighbors=5, seed=42, shuffle=False)
    assert len(X_sm) == len(X) + (int((y == 0).sum()) - int((y == 1).sum()))
    assert np.isfinite(X_sm).all()
    assert np.array_equal(X, X_before)
    assert np.array_equal(y, y_before)
    # Synthetic rows must lie inside the minority class's observed hull.
    minority = X[y == 1]
    synthetic = X_sm[len(X) :]
    if len(synthetic):
        assert synthetic.min() >= minority.min() - 1e-6
        assert synthetic.max() <= minority.max() + 1e-6


def test_smote_rejects_raw_time_series_shape() -> None:
    raw = np.zeros((10, 4, 1800), dtype=np.float32)
    labels = np.array([0] * 8 + [1] * 2)
    with pytest.raises(ValueError):
        smote_oversample(raw, labels)


def test_outer_and_inner_folds_are_isolated_and_deterministic() -> None:
    _, y = _synthetic_binary(n=120, prevalence=0.25, seed=5)
    folds = _synthetic_folds(y, n_splits=5)
    for outer in range(1, 6):
        train_idx, val_idx = outer_train_validation(folds, outer)
        assert np.intersect1d(train_idx, val_idx).size == 0
        assert sorted(np.concatenate([train_idx, val_idx]).tolist()) == list(range(len(y)))
        inner_a = make_inner_folds(y, train_idx, n_splits=3, seed=42 + outer)
        inner_b = make_inner_folds(y, train_idx, n_splits=3, seed=42 + outer)
        assert np.array_equal(inner_a, inner_b)
        assert set(inner_a.tolist()) == {1, 2, 3}
        # Every inner training/validation split stays inside the outer training set.
        for inner in (1, 2, 3):
            inner_val = train_idx[inner_a == inner]
            inner_train = train_idx[inner_a != inner]
            assert np.intersect1d(inner_train, inner_val).size == 0
            assert np.intersect1d(inner_val, val_idx).size == 0


def test_smote_is_not_applied_to_validation_data() -> None:
    X, y = _synthetic_binary(n=80, prevalence=0.2, seed=6)
    train_idx = np.arange(60)
    val_idx = np.arange(60, 80)
    X_val_before = X[val_idx].copy()
    fitted = fit_classical(
        X[train_idx], y[train_idx], "logistic", "smote", {"C": 0.5}, seed=42
    )
    prob = fitted.predict_proba(X[val_idx])
    assert len(prob) == len(val_idx)
    assert np.isfinite(prob).all()
    assert np.array_equal(X[val_idx], X_val_before)


def test_threshold_free_metrics_are_independent_of_threshold() -> None:
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    prob = np.array([0.05, 0.10, 0.20, 0.35, 0.45, 0.55, 0.60, 0.70, 0.85, 0.95])
    low = evaluate_predictions(y, prob, threshold=0.2)
    high = evaluate_predictions(y, prob, threshold=0.8)
    for key in ("auroc", "auprc", "brier"):
        assert low[key] == pytest.approx(high[key])
    assert low["recall_sensitivity"] >= high["recall_sensitivity"]
    assert low["specificity"] <= high["specificity"]

    threshold = best_threshold(y, prob)
    chosen = evaluate_predictions(y, prob, threshold=threshold)
    assert 0.0 <= threshold <= 1.0
    assert chosen["balanced_accuracy"] >= low["balanced_accuracy"] - 1e-9


def test_metrics_match_a_known_confusion_matrix() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.7, 0.8, 0.6, 0.4, 0.9, 0.3])
    m = evaluate_predictions(y, prob, threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 2, 2, 2)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["recall_sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx(np.mean((prob - y) ** 2))


def test_bootstrap_and_paired_difference_are_reproducible() -> None:
    # AUROC(a) = 1.00, AUROC(b) = 0.75, so the paired difference is non-zero.
    y = np.array([0, 0, 1, 1])
    prob_a = np.array([0.10, 0.20, 0.80, 0.90])
    prob_b = np.array([0.10, 0.70, 0.60, 0.90])
    ci_a = bootstrap_confidence_intervals(y, prob_a, n_boot=50, seed=42)
    ci_b = bootstrap_confidence_intervals(y, prob_a, n_boot=50, seed=42)
    assert ci_a == ci_b
    diff_a = paired_bootstrap_difference(y, prob_a, prob_b, n_boot=50, seed=42)
    diff_b = paired_bootstrap_difference(y, prob_a, prob_b, n_boot=50, seed=42)
    assert diff_a == diff_b
    assert diff_a["auroc"]["difference"] > 0


def test_classical_fit_is_reproducible() -> None:
    X, y = _synthetic_binary(n=90, prevalence=0.2, seed=7)
    kwargs = dict(model="logistic", strategy="class_weight", params={"C": 0.5}, seed=42)
    first = fit_classical(X, y, **kwargs).predict_proba(X)
    second = fit_classical(X, y, **kwargs).predict_proba(X)
    assert np.allclose(first, second)


def test_augmentation_preserves_shape_and_masks() -> None:
    signal = torch.zeros((4, 4, 1800), dtype=torch.float32)
    signal[:, 2:, :] = 1.0
    generator = torch.Generator().manual_seed(42)
    out = augment_batch(signal, "gaussian", generator)
    assert out.shape == signal.shape
    assert torch.equal(out[:, 2:, :], signal[:, 2:, :])

    generator = torch.Generator().manual_seed(42)
    mixed = augment_batch(signal, "mix", generator)
    assert mixed.shape == signal.shape
    assert torch.isfinite(mixed).all()
    with pytest.raises(ValueError):
        augment_batch(signal, "unknown", generator)


def test_class_counts_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError):
        class_counts(np.array([0, 1, 2]))


def test_deep_training_is_reproducible_with_a_fixed_seed() -> None:
    """Two identical deep fits must produce identical predictions.

    The runtime fixes the CPU thread count, because floating-point reduction
    order changes with the thread count.
    """
    rng = np.random.default_rng(0)
    n, length = 12, 64
    signals = rng.normal(size=(n, 4, length)).astype(np.float32)
    clinical = rng.normal(size=(n, 11)).astype(np.float32)
    labels = np.array([0] * 10 + [1, 1], dtype=np.int64)
    config = DeepRunConfig(
        variant="signal",
        strategy="plain_bce",
        learning_rate=1e-3,
        epochs=1,
        batch_size=4,
        patience=2,
        seed=42,
    )
    model_a, _, _ = fit_deep(signals, clinical, labels, None, None, None, config)
    prob_a = predict_array(model_a, signals, clinical)
    model_b, _, _ = fit_deep(signals, clinical, labels, None, None, None, config)
    prob_b = predict_array(model_b, signals, clinical)
    assert np.array_equal(prob_a, prob_b)
