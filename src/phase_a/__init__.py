"""Phase A: data handling and preprocessing optimisation experiments.

This package is deliberately additive: the production preprocessing,
training, API and frontend are left untouched.  Phase A uses the same raw
recordings, labels, split file and model architectures, and writes every
artefact to ``experiments/phase_a/``.
"""

__all__ = ["config", "quality", "signals", "features", "selection", "experiment"]
