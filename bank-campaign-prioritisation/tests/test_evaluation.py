import numpy as np
import pytest

from src.evaluation import evaluate_ranking


def test_hand_calculated_metrics():
    result = evaluate_ranking(
        target=[1, 0, 1, 0, 0],
        probabilities=[0.9, 0.8, 0.7, 0.2, 0.1],
        capacity=0.40,
    )

    assert result["selected"] == 2
    assert result["captured"] == 1
    assert result["precision"] == pytest.approx(0.50)
    assert result["recall"] == pytest.approx(0.50)
    assert result["lift"] == pytest.approx(1.25)
    assert result["brier"] == pytest.approx(0.158)


def test_budget_is_not_exceeded():
    result = evaluate_ranking(
        [1, 0, 0], [0.9, 0.2, 0.1], capacity=0.10
    )

    assert result["selected"] == 0
    assert result["captured"] == 0
    assert np.isnan(result["precision"])


def test_ties_preserve_input_order():
    result = evaluate_ranking(
        [1, 0], [0.5, 0.5], capacity=0.50
    )

    assert result["captured"] == 1


def test_no_subscribers():
    result = evaluate_ranking(
        [0, 0], [0.8, 0.2], capacity=0.50
    )

    assert result["precision"] == 0
    assert np.isnan(result["recall"])
    assert np.isnan(result["lift"])


def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError):
        evaluate_ranking([1, 0], [1.2, 0.1])