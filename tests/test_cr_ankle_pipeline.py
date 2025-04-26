import numpy as np
import pytest
from code.cr_ankle_pipeline import single_power, fit_single

def test_single_power():
    E = np.array([1.0, 2.0, 4.0])
    A, gamma = 2.0, 1.0
    expected = A * (E ** (-gamma))
    np.testing.assert_allclose(single_power(E, A, gamma), expected)

def test_fit_single():
    E = np.array([1.0, 2.0, 4.0])
    A_true, gamma_true = 3.0, 2.0
    J = A_true * E ** (-gamma_true)
    dJ = np.ones_like(J)
    p0 = (1.0, 1.0)
    params, _ = fit_single(E, J, dJ, p0)
    assert pytest.approx(A_true, rel=1e-6) == params[0]
    assert pytest.approx(gamma_true, rel=1e-6) == params[1]
