import numpy as np
from sklearn.metrics import brier_score_loss


def evaluate_ranking(target, probabilities, capacity=0.10):
    target = np.asarray(target)
    probabilities = np.asarray(probabilities, dtype=float)

    if target.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(target) == 0 or len(target) != len(probabilities):
        raise ValueError("Inputs must have the same nonzero length.")

    if not np.isin(target, [0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and between 0 and 1.")

    if not np.isfinite(capacity) or not 0 <= capacity <= 1:
        raise ValueError("Capacity must be between 0 and 1.")

    count = int(np.floor(len(target) * capacity))
    selected = np.argsort(
        -probabilities, kind="stable"
    )[:count]

    captured = int(target[selected].sum())
    total_subscribers = int(target.sum())
    base_rate = target.mean()

    precision = captured / count if count else np.nan

    return {
        "selected": count,
        "captured": captured,
        "precision": precision,
        "recall": (
            captured / total_subscribers
            if total_subscribers else np.nan
        ),
        "lift": precision / base_rate if base_rate else np.nan,
        "brier": brier_score_loss(target, probabilities),
    }