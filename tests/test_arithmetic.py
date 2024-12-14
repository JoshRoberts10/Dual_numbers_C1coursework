# This file covers basic tests for arithmetic operations of the Dual class
import pytest
import numpy as np
from dual_autodiff.dual import Dual




def test_addition():
    """
    Tests the functionality of the addition operator of our dual
    
    """
    # dual plus dual 
    d1 = Dual(1,1)
    d2 = Dual(2,2)
    x = d1+d2
    assert x.real == pytest.approx(3, rel = 1e-12)
    assert x.dual == pytest.approx(3, rel = 1e-12)

    # dual plus scalar 
    d1 = Dual(1,1)
    d2 = 1
    x = d1+d2
    assert x.real == pytest.approx(2, rel = 1e-12)
    assert x.dual == pytest.approx(1, rel = 1e-12)

    # Scalar plus Dual 
    d1 = 1
    d2 = Dual(1,1)
    x = d1+d2
    assert x.real == pytest.approx(2, rel = 1e-12)
    assert x.dual == pytest.approx(1, rel = 1e-12)

    # dual plus dual floats 
    d1 = Dual(1.5,1.2)
    d2 = Dual(2.2,2.1)
    x = d1+d2
    assert x.real == pytest.approx(3.7, rel = 1e-12)
    assert x.dual == pytest.approx(3.3, rel = 1e-12)


    # integer plus dual floats 
    d1 = Dual(1, 1)
    d2 = Dual(2.2, 2.1)
    x = d1+d2
    assert x.real == pytest.approx(3.2, rel = 1e-12)
    assert x.dual == pytest.approx(3.1, rel = 1e-12)
    
    # dual plus negative dual
    d1 = Dual(1,1)
    d2 = Dual(-2, -2)
    x = d1+d2
    assert x.real == pytest.approx(-1, rel = 1e-12)
    assert x.dual == pytest.approx(-1, rel = 1e-12)

    # edge case 0,0 
    d1 = Dual(0,0)
    d2 = Dual(-2, -2)
    x = d1+d2
    assert x.real == pytest.approx(-2, rel = 1e-12)
    assert x.dual == pytest.approx(-2, rel = 1e-12)

    # edge case 0,0 
    d1 = Dual(0,0)
    d2 = Dual(0, 0)
    x = d1+d2
    assert x.real == pytest.approx(0, rel = 1e-12)
    assert x.dual == pytest.approx(0, rel = 1e-12)

    # edge case dual plus incorrect type
    with pytest.raises(TypeError):
        Dual(1,1) + "string"
    # edge case dual plus incorrect type
    with pytest.raises(TypeError):
        Dual(1,1) + None
    
    # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         Dual(1,1) + np.inf
    # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         Dual(1,1) + np.nan


    # # edge case dual plus incorrect type
    with pytest.raises(TypeError):
         "string" + Dual(1,1)
    # # edge case dual plus incorrect type
    with pytest.raises(TypeError):
         None + Dual(1,1)
    
    # # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         np.inf + Dual(1,1)
    # # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         np.nan + Dual(1,1)


def test_subtraction():
    """
    Tests the functionality of the subtraction operator of our dual
    
    """
    # dual minus dual 
    d1 = Dual(1,1)
    d2 = Dual(2,2)
    x = d1-d2
    assert x.real == pytest.approx(-1, rel = 1e-12)
    assert x.dual == pytest.approx(-1, rel = 1e-12)

    # dual minus scalar 
    d1 = Dual(1,1)
    d2 = 1
    x = d1-d2
    assert x.real == pytest.approx(0, rel = 1e-12)
    assert x.dual == pytest.approx(1, rel = 1e-12)

    # Scalar minus Dual 
    d1 = 1
    d2 = Dual(1,1)
    x = d1-d2
    assert x.real == pytest.approx(0, rel = 1e-12)
    assert x.dual == pytest.approx(-1, rel = 1e-12)

    # dual minus dual floats 
    d1 = Dual(1.5,1.2)
    d2 = Dual(2.2,2.1)
    x = d1-d2
    assert x.real == pytest.approx(-0.7, rel = 1e-12)
    assert x.dual == pytest.approx(-0.9, rel = 1e-12)

    # integer minus dual floats 
    d1 = Dual(1, 1)
    d2 = Dual(2.2, 2.1)
    x = d1-d2
    assert x.real == pytest.approx(-1.2, rel = 1e-12)
    assert x.dual == pytest.approx(-1.1, rel = 1e-12)
    
    # dual minus negative dual
    d1 = Dual(1,1)
    d2 = Dual(-2, -2)
    x = d1-d2
    assert x.real == pytest.approx(3, rel = 1e-12)
    assert x.dual == pytest.approx(3, rel = 1e-12)

    # edge case 0,0 
    d1 = Dual(0,0)
    d2 = Dual(-2, -2)
    x = d1-d2
    assert x.real == pytest.approx(2, rel = 1e-12)
    assert x.dual == pytest.approx(2, rel = 1e-12)

    # edge case 0,0 
    d1 = Dual(0,0)
    d2 = Dual(0, 0)
    x = d1-d2
    assert x.real == pytest.approx(0, rel = 1e-12)
    assert x.dual == pytest.approx(0, rel = 1e-12)

    # edge case dual plus incorrect type
    with pytest.raises(TypeError):
        Dual(1,1) - "string"
    # edge case dual plus incorrect type
    with pytest.raises(TypeError):
        Dual(1,1) - None
    
    # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         Dual(1,1) - np.inf
    # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         Dual(1,1) - np.nan


    # # edge case dual plus incorrect type
    with pytest.raises(TypeError):
         "string" - Dual(1,1)
    # # edge case dual plus incorrect type
    with pytest.raises(TypeError):
         None - Dual(1,1)
    
    # # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         np.inf - Dual(1,1)
    # # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         np.nan - Dual(1,1)

def test_multiplication():
    """
    Tests the functionality of the multiplication operator of our dual class

    """

    # dual times dual 
    d1 = Dual(1,1)
    d2 = Dual(2,2)
    x = d1*d2
    assert x.real == pytest.approx(2, 1e-12)
    assert x.dual == pytest.approx(4, rel = 1e-12)

    # dual times scalar 
    d1 = Dual(1,1)
    d2 = 1
    x = d1*d2
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(1, 1e-12)

    # Scalar times Dual 
    d1 = 1
    d2 = Dual(1,1)
    x = d1*d2
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(1, 1e-12)

    # dual times dual floats 
    d1 = Dual(1.5,1.2)
    d2 = Dual(2.2,2.1)
    x = d1*d2
    assert x.real == pytest.approx(3.3, 1e-12)
    assert x.dual == pytest.approx(5.79, 1e-12)

    # integer times dual floats 
    d1 = Dual(1, 1)
    d2 = Dual(2.2, 2.1)
    x = d1*d2
    assert x.real == pytest.approx(2.2, 1e-12)
    assert x.dual == pytest.approx(4.3, 1e-12)
    
    # dual times negative dual
    d1 = Dual(1,1)
    d2 = Dual(-2, -2)
    x = d1*d2
    assert x.real == pytest.approx(-2, 1e-12)
    assert x.dual == pytest.approx(-4, 1e-12)

    # edge case 0,0 
    d1 = Dual(0,0)
    d2 = Dual(-2, -2)
    x = d1*d2
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)

    # edge case 0,0 
    d1 = Dual(0,0)
    d2 = Dual(0, 0)
    x = d1*d2
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)


    #check commutitivity
    d1 = Dual(3,4)
    d2 = Dual(10,12)
    x = d1*d2
    y = d2*d1
    assert x.real == pytest.approx(y.real, 1e-12)
    assert x.dual == pytest.approx(y.dual, 1e-12)

    #checks associativity
    d1 = Dual(3,4)
    d2 = Dual(10,12)
    d3 = Dual(31, 1.5)
    x = d1*(d2*d3)
    y = (d1*d2)*d3
    assert x.real == pytest.approx(y.real, 1e-12)
    assert x.dual == pytest.approx(y.dual, 1e-12)
    
    # edge case dual plus incorrect type
    with pytest.raises(TypeError):
        Dual(1,1) * "string"
    # edge case dual plus incorrect type
    with pytest.raises(TypeError):
        Dual(1,1) * None
    
    # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         Dual(1,1) * np.inf
    # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         Dual(1,1) * np.nan

    # # edge case dual plus incorrect type
    with pytest.raises(TypeError):
         "string" * Dual(1,1)
    # # edge case dual plus incorrect type
    with pytest.raises(TypeError):
         None * Dual(1,1)
    
    # # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         np.inf * Dual(1,1)
    # # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         np.nan * Dual(1,1)



def test_division():
    """
    Tests the functionality of the division operator of our dual class

    """

    # dual times dual 
    d1 = Dual(1,1)
    d2 = Dual(2,2)
    x = d1/d2
    assert x.real == pytest.approx(0.5, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)

    # dual times scalar 
    d1 = Dual(1,1)
    d2 = 1
    x = d1/d2
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(1, 1e-12)

    # Scalar times Dual 
    d1 = 1
    d2 = Dual(1,1)
    x = d1/d2
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(-1, 1e-12)

    # dual times dual floats 
    d1 = Dual(1.5,1.2)
    d2 = Dual(2.2,2.1)
    x = d1/d2
    assert x.real == pytest.approx(0.6818181818181818, 1e-12)
    assert x.dual == pytest.approx(-0.10537190082644632, 1e-12)

    # integer times dual floats 
    d1 = Dual(1, 1)
    d2 = Dual(2.2, 2.1)
    x = d1/d2
    assert x.real == pytest.approx(0.45454545454545453, 1e-12)
    assert x.dual == pytest.approx(0.020661157024793403, 1e-12)

    # dual times negative dual
    d1 = Dual(1,1)
    d2 = Dual(-2, -2)
    x = d1/d2
    assert x.real == pytest.approx(-0.5, 1e-12)
    assert x.dual == pytest.approx(-0, 1e-12)

    # edge case 0,0 
    d1 = Dual(0,0)
    d2 = Dual(-2, -2)
    x = d1/d2
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)

    # edge case 0,0 
    d1 = Dual(0,0)
    d2 = Dual(0, 0)
    with pytest.raises(ZeroDivisionError):
        x = d1/d2

    # edge case dual plus incorrect type
    with pytest.raises(TypeError):
        Dual(1,1) / "string"
    # edge case dual plus incorrect type
    with pytest.raises(TypeError):
        Dual(1,1)/  None
    
    # dividing by infinity should return 0
    d1 = Dual(2,2)
    d2 = np.inf
    x = d1/d2
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)

    d1 = Dual(2, 2)
    x = d1 / -np.inf
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)


    #testing associtivuty
    d1 = Dual(1.5,1.2)
    d2 = Dual(2.2,2.1)
    d3 = Dual(2,2)
    x = d1/d2/d3
    y = d1/(d2*d3)
    assert x.real == pytest.approx(y.real, 1e-12)
    assert x.dual == pytest.approx(y.dual, 1e-12)

    # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         Dual(1,1) / np.nan

    # # edge case dual plus incorrect type
    with pytest.raises(TypeError):
         "string" / Dual(1,1)
    # # edge case dual plus incorrect type
    with pytest.raises(TypeError):
         None / Dual(1,1)
    
    # # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         np.inf / Dual(1,1)
    # # edge case dual plus incorrect type
    with pytest.raises(ValueError):
         np.nan / Dual(1,1)






