import pytest
import numpy as np
from dual_autodiff.dual import Dual

def test_dual_initialisation_valid():
    """
    Runs test to ensure the Dual class can be initialised correctly when ran with valid imports
    
    """

    # simple initialisation with 2 integers
    d = Dual(1,2)
    assert d.real == 1
    assert d.dual == 2
    
    # negatives test
    d = Dual(-1,-2)
    assert d.real == -1
    assert d.dual == -2

    # no dual
    d = Dual(-1,0)
    assert d.real == -1
    assert d.dual == 0 

    # no real 
    d = Dual(0,1)
    assert d.real == 0
    assert d.dual == 1

    # both zero  
    d = Dual(0,0)
    assert d.real == 0
    assert d.dual == 0  

    # floats  
    d = Dual(2.2,3.4)
    assert d.real == 2.2
    assert d.dual == 3.4  

    #really big and small numbers
    d = Dual(1e10, 1e-10)
    assert d.real == 1e10
    assert d.dual == 1e-10 

    #mix of floats and integers
    d = Dual(1, 3.4)
    assert d.real == 1
    assert d.dual == 3.4



def test_dual_initialisation_invalid():
    """
    Tests the cases when my initialisation should not work
    
    """


    # lets test using strings for either real or dual or both 
    with pytest.raises(TypeError, match = "real component must be either a float or an integer"):
        Dual('real string', 1)

    with pytest.raises(TypeError, match = "dual component must be either a float or an integer"):
        Dual(1, "dual string")

    with pytest.raises(TypeError):
        Dual("real string", "dual string")

    # lets test if we put None in 
    with pytest.raises(TypeError, match = "real component must be either a float or an integer"):
        Dual(None, 1)

    with pytest.raises(TypeError, match = "dual component must be either a float or an integer"):
        Dual(1, None)

    with pytest.raises(TypeError):
        Dual(None, None)



    # lets test if we put dual in 
    with pytest.raises(TypeError, match = "real component must be either a float or an integer"):
        Dual(Dual(1,1), 1)

    with pytest.raises(TypeError, match = "dual component must be either a float or an integer"):
        Dual(1, Dual(1,1))

    with pytest.raises(TypeError):
        Dual(Dual(1,1), Dual(1,1))


    #  lets try using infinite and nan values
    with pytest.raises(ValueError, match = "real component cannot be nan"):
        Dual(np.nan, 1)

    with pytest.raises(ValueError, match = "dual component cannot be nan"):
         Dual(1, np.nan)

    with pytest.raises(ValueError):
         Dual(np.nan, np.nan)

    with pytest.raises(ValueError):
        Dual(np.inf, 1)

    with pytest.raises((ValueError)):
        Dual(1, np.inf)

    with pytest.raises((ValueError)):
        Dual(np.inf, np.inf)


def test_comparison():
    """
    Testing the correct functionality of the comparison method
    """

    #standard test 

    d1 = Dual(5.0, -2.0)
    d2 = Dual(5.0, -2.0)
    assert d1 == d2

    d1 = Dual(5.0, -10.0)
    d2 = Dual(5.0, -2.0)
    assert not d1 == d2

    d1 = Dual(6.0, -2.0)
    d2 = Dual(5.0, -2.0)
    assert not d1 == d2

    d1 = Dual(3.0, 0.0)
    assert d1 == 3.0

    d1 = Dual(3.0, 0.0)
    assert 3.0 == d1

    d1 = Dual(3.0, 0.1)
    assert not d1 == 3.0

    #testing within tolerance
    d1 = Dual(3.0, 1.000000000000001)  
    d2 = Dual(3.0, 1.0)
    assert d1 == d2

    d1 = Dual(0, 0)  
    d2 = Dual(0, 0)
    assert d1 == d2

    d1 = Dual(0, 0)  
    d2 = 'Dual'
    with pytest.raises(TypeError):
        assert d1 == d2

    d1 = Dual(0, 0) 
    d2 = 'Dual'
    with pytest.raises(TypeError):
        assert d2 == d1




    

