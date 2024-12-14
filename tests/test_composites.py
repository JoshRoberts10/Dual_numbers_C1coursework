import pytest
import numpy as np
from dual_autodiff.dual import Dual





########## please note the following tests below were generated using generative AI (unlike the other test files) I take no ######
########## credit for the below tests, I included as wanted to be exhaustitive in testing yet its somewhat tedious writing  ######
########## out even more testing again for all my functions. I instructed chatgpt with the prompt:
########## " Please provide testing of the combinations of functions for my Dual class, including addition, subraction, multiplicatin
##########  division, powers, logarithms, exponentials trigonemtric functions and hypervolic functions."
##########   I then pasted an example of one of my other tests    #######



def test_composite_arithmetic():
    d1 = Dual(2, 3)
    d2 = Dual(4, -1)
    d3 = Dual(0.5, 2)

    # Perform a series of operations
    result = (d1 + d2) * d3 - d2 / d1


    
    assert result.real == pytest.approx(1, rel=1e-12)
    assert result.dual == pytest.approx(16.5, rel=1e-12)



def test_composite_trigonometric():
    d = Dual(np.pi / 4, 1)

    # Perform a composite operation
    result = d.sin() + d.cos() * d.tan()

    # Expected values
    sin_real = np.sin(np.pi / 4)
    sin_dual = np.cos(np.pi / 4)
    cos_real = np.cos(np.pi / 4)
    cos_dual = -np.sin(np.pi / 4)
    tan_real = np.tan(np.pi / 4)
    tan_dual = 1 / (np.cos(np.pi / 4) ** 2)

    real_part = sin_real + cos_real * tan_real
    dual_part = sin_dual + cos_dual * tan_real + cos_real * tan_dual

    assert result.real == pytest.approx(real_part, rel=1e-12)
    assert result.dual == pytest.approx(dual_part, rel=1e-12)


def test_exponential_log_identity():
    d = Dual(2, 3)

    # Composite operation
    result = d.log().exp()

    # Should return the original dual number
    assert result.real == pytest.approx(d.real, rel=1e-12)
    assert result.dual == pytest.approx(d.dual, rel=1e-12)


def test_composite_hyperbolic_trigonometric():
    d = Dual(1, 0.5)

    # Composite operation
    result = d.sinh() + d.cosh() * d.tanh()

    # Expected values
    sinh_real = np.sinh(1)
    sinh_dual = np.cosh(1) * 0.5
    cosh_real = np.cosh(1)
    cosh_dual = np.sinh(1) * 0.5
    tanh_real = np.tanh(1)
    tanh_dual = 0.5 / (np.cosh(1) ** 2)

    real_part = sinh_real + cosh_real * tanh_real
    dual_part = sinh_dual + cosh_dual * tanh_real + cosh_real * tanh_dual

    assert result.real == pytest.approx(real_part, rel=1e-12)
    assert result.dual == pytest.approx(dual_part, rel=1e-12)


def test_composite_power_log_sin():
    d = Dual(2, 0.1)

    # Composite operation
    result = (d ** 2).log() + d.sin()

    # Expected values
    power_real = 2 ** 2
    power_dual = 2 * 2 * 0.1
    log_real = np.log(power_real)
    log_dual = power_dual / power_real
    sin_real = np.sin(2)
    sin_dual = np.cos(2) * 0.1

    real_part = log_real + sin_real
    dual_part = log_dual + sin_dual

    assert result.real == pytest.approx(real_part, rel=1e-12)
    assert result.dual == pytest.approx(dual_part, rel=1e-12)

