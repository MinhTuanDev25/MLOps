from __future__ import annotations

from churn_mldevops.monitoring import psi


def test_psi_identical_distributions_low() -> None:
    p = [0.2, 0.3, 0.5]
    assert psi(p, p) < 1e-5


def test_psi_shifted_distribution_positive() -> None:
    expected = [0.25, 0.25, 0.25, 0.25]
    actual = [0.05, 0.05, 0.05, 0.85]
    val = psi(expected, actual)
    assert val > 0.1
