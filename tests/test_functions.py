import pytest
import numpy as np
from dual_autodiff.dual import Dual


def test_log():

    """
    Tests the log method in the Dual class
    
    """

    #normal test case
    d = Dual(2,3)
    x = d.log()
    assert x.real == pytest.approx(0.6931471805599453, 1e-12)
    assert x.dual == pytest.approx(1.5, 1e-12)

    #floats
    d = Dual(10.234, 40.332)
    x = d.log()
    assert x.real == pytest.approx(2.3257155103829, 1e-12)
    assert x.dual == pytest.approx(3.940981043580223, 1e-12)

    #negative dual
    d = Dual(2, -4)
    x = d.log()
    assert x.real == pytest.approx(0.6931471805599453, 1e-12)
    assert x.dual == pytest.approx(-2, 1e-12)

    #small real 
    d = Dual(1e-10, 1)
    x = d.log()
    assert x.real == pytest.approx(-23.025850929940457, 1e-12)
    assert x.dual == pytest.approx(10000000000, 1e-12)

    #neg real
    with pytest.raises(ValueError):
        Dual(-2,3).log()

    #0 real
    with pytest.raises(ValueError):
        Dual(0,3).log()
    

def test_exponential():
    """
    Test the exponential part of our Dual class
    """

    #normal test case
    d = Dual(2,3)
    x = d.exp()
    assert x.real == pytest.approx(7.38905609893065, 1e-12)
    assert x.dual == pytest.approx(22.16716829679195, 1e-12)

    #floats
    d = Dual(10.234, 40.332)
    x = d.exp()
    assert x.real == pytest.approx(27833.62218441052, 1e-12)
    assert x.dual == pytest.approx(1122585.649941645, 1e-12)

    #negative dual
    d = Dual(2, -4)
    x = d.exp()
    assert x.real == pytest.approx(7.38905609893065, 1e-12)
    assert x.dual == pytest.approx(-29.5562243957226, 1e-12)

    d = Dual(0, 1)
    x = d.exp()
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(1, 1e-12)

    d = Dual(1, 0)
    x = d.exp()
    assert x.real == pytest.approx(2.718281828459045, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)

    #small real 
    d = Dual(1e-10, 1)
    x = d.exp()
    assert x.real == pytest.approx(1.0000000001, 1e-12)
    assert x.dual == pytest.approx(1.0000000001, 1e-12)



def test_pow():
    """
    Tests the power operation in my Dual class
    
    """

    # first lets focus on different cases then we can move to edge cases

    #test scaler to dual and dual to scaler



    # dual to scaler
    d1 = Dual(2,3)
    d2 = 2
    x = d1**d2
    assert x.real == pytest.approx(4, 1e-12)
    assert x.dual == pytest.approx(12, 1e-12)

    #ensure capatiability between differnt objects
    d1 = Dual(2,3)
    d2 = Dual(2,0)
    x = d1**d2
    assert x.real == pytest.approx(4, 1e-12)
    assert x.dual == pytest.approx(12, 1e-12)

    # dual to float
    d1 = Dual(2,3)
    d2 = 2.5
    x = d1**d2
    assert x.real == pytest.approx(5.656854249492381, 1e-12)
    assert x.dual == pytest.approx(21.213203435596427, 1e-12)

    # dual to fraction
    d1 = Dual(2,3)
    d2 = 0.5
    x = d1**d2
    assert x.real == pytest.approx(1.4142135623730951, 1e-12)
    assert x.dual == pytest.approx(1.0606601717798214, 1e-12)


    # Scalar raised to Dual
    d = Dual(2, 1)
    x = 3 ** d
    assert x.real == pytest.approx(9, 1e-12)  
    assert x.dual == pytest.approx(9.88751059801298, 1e-12)  


    #Dual to Dual
    d1 = Dual(1,2)
    d2 = Dual(3,4)
    x = d1**d2
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(6, 1e-12)

    #Dual to Dual with floats
    d1 = Dual(1.5,2.2)
    d2 = Dual(3.1,4.8)
    x = d1**d2
    assert x.real == pytest.approx(3.514656635974386, 1e-12)
    assert x.dual == pytest.approx(22.82031120933202, 1e-12)

    # sense check
    d1 = Dual(2, 0)
    d2 = Dual(2, 0)
    x = d1**d2
    assert x.real == pytest.approx(4, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)




    # Now lets consider some edge cases

    # first lets look at edge cases for scalar raised to dual 

    # 0 raised to a dual 
    with pytest.raises(ValueError):
        0**Dual(1,1)

    # check consistancy accros implementation 
    with pytest.raises(ValueError):
        Dual(0,0)**Dual(1,1)

    
    # check negative scalar
    with pytest.raises(ValueError) as excinfo:
        (-2)**Dual(1,1)

    # check consistancy accros implementation 
    with pytest.raises(ValueError):
        Dual(-2,0)**Dual(1,1)

    
    #now consider edge cases for dual raised to scalar

    # negative real to fractional power should return Value error
    with pytest.raises(ValueError):
        Dual(-2,3)**(-1/2)

    # 0 real to 0 power in dual, could possibly implement this differntly, however im happy with logic currently
    with pytest.raises(ValueError):
        Dual(0,3)**1 

    # should give 0^0 error 
    with pytest.raises(ValueError):
        Dual(0,3)**0 

    
    #now lets check dual to dual

    # negative real base with none zero powers should give error
    with pytest.raises(ValueError):
        d1 = Dual(-2,3)
        d2 = Dual(3,4)
        d1**d2


    # 0 real base with zero real power results in 0^0 and should thus give valeu error
    with pytest.raises(ValueError):
        d1 = Dual(0,3)
        d2 = Dual(0,4)
        d1**d2

    # 0 real base with negative real power results in zero division and should thus give valeu error
    with pytest.raises(ValueError):
        d1 = Dual(0,3)
        d2 = Dual(-2,4)
        d1**d2
    
    # 0 real base with real power 1 results in zero division and should thus give valeu error
    with pytest.raises(ValueError):
        d1 = Dual(0,3)
        d2 = Dual(1,4)
        d1**d2


    # dual raised to dual with zero dual component
    d = Dual(2, 0)
    x = d ** 0
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)

    

def test_trig():
    """
    Testing standard trigonometric functions
    """

    # test some standard cases sines
    d = Dual(np.pi, 2)
    x = d.sin()
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(-2, 1e-12)

    d = Dual(2.123, 3.45)
    x = d.sin()
    assert x.real == pytest.approx(0.8513706211471594, 1e-12)
    assert x.dual == pytest.approx(-1.809748020855077, 1e-12)

    d = Dual(0,1)
    x = d.sin()
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(1, 1e-12)
    
    d = Dual(np.pi / 4, 1)
    x = d.sin()
    assert x.real == pytest.approx(0.7071067811865476, 1e-12) 
    assert x.dual == pytest.approx(0.7071067811865476, 1e-12)  

    d = Dual(0, 0)
    x = d.sin()
    assert x.real == pytest.approx(0, 1e-12) 
    assert x.dual == pytest.approx(0, 1e-12)  


    d = Dual(-np.pi / 4, 1)
    x = d.sin()
    assert x.real == pytest.approx(-0.7071067811865476, 1e-12) 
    assert x.dual == pytest.approx(0.7071067811865476, 1e-12)  


    # not cases of incorrect types are all handled by the instance method and are tested there


    # test some standard cases for cosines
    d = Dual(np.pi, 2)
    x = d.cos()
    assert x.real == pytest.approx(-1, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)

    d = Dual(2.123, 3.45)
    x = d.cos()
    assert x.real == pytest.approx(-0.5245646437261092, 1e-12)
    assert x.dual == pytest.approx(-2.9372286429577, 1e-12)

    d = Dual(0,1)
    x = d.cos()
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)
    
    d = Dual(np.pi / 4, 1)
    x = d.cos()
    assert x.real == pytest.approx(0.7071067811865476, 1e-12) 
    assert x.dual == pytest.approx(-0.7071067811865476, 1e-12)  

    d = Dual(0, 0)
    x = d.cos()
    assert x.real == pytest.approx(1, 1e-12) 
    assert x.dual == pytest.approx(0, 1e-12)  


    d = Dual(-np.pi / 4, 1)
    x = d.cos()
    assert x.real == pytest.approx(0.7071067811865476, 1e-12) 
    assert x.dual == pytest.approx(0.7071067811865476, 1e-12)  


    #now lets tets tans

    d = Dual(np.pi/4, 2)
    x = d.tan()
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(4, 1e-12)

    d = Dual(2.123, 3.45)
    x = d.tan()
    assert x.real == pytest.approx(-1.623004202303206, 1e-12)
    assert x.dual == pytest.approx(12.5377921103938414, 1e-12)


    d = Dual(0,1)
    x = d.tan()
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(1, 1e-12)
    
    d = Dual(np.pi / 4, 1)
    x = d.tan()
    assert x.real == pytest.approx(1, 1e-12) 
    assert x.dual == pytest.approx(2, 1e-12)  

    d = Dual(0, 0)
    x = d.tan()
    assert x.real == pytest.approx(0, 1e-12) 
    assert x.dual == pytest.approx(0, 1e-12)  


    d = Dual(-np.pi / 4, 1)
    x = d.tan()
    assert x.real == pytest.approx(-1, 1e-12) 
    assert x.dual == pytest.approx(2, 1e-12)  

    #test undefined point
    d = Dual(np.pi / 2, 1)
    with pytest.raises(ZeroDivisionError):
        d.tan()


def test_hyp_trig():
    """
    testing the hyperbolic trig functons
    
    """

    # test some standard cases sines
    d = Dual(np.pi, 2)
    x = d.sinh()
    assert x.real == pytest.approx(11.548739357257746, 1e-12)
    assert x.dual == pytest.approx(23.183906551043037, 1e-12)

    d = Dual(2.123, 3.45)
    x = d.sinh()
    assert x.real == pytest.approx(4.118248177279596, 1e-12)
    assert x.dual == pytest.approx(14.62082486425297, 1e-12)

    d = Dual(0,1)
    x = d.sinh()
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(1, 1e-12)
    
    d = Dual(np.pi / 4, 1)
    x = d.sinh()
    assert x.real == pytest.approx(0.8686709614860095, 1e-12) 
    assert x.dual == pytest.approx(1.324609089252006, 1e-12)  

    d = Dual(0, 0)
    x = d.sinh()
    assert x.real == pytest.approx(0, 1e-12) 
    assert x.dual == pytest.approx(0, 1e-12)  


    d = Dual(-np.pi / 4, 1)
    x = d.sinh()
    assert x.real == pytest.approx(-0.8686709614860095, 1e-12) 
    assert x.dual == pytest.approx(1.324609089252006, 1e-12)  


    # note that cases of incorrect types are all handled by the instance method and are tested there


    # test some standard cases for cosines
    d = Dual(np.pi, 2)
    x = d.cosh()
    assert x.real == pytest.approx(11.591953275521519, 1e-12)
    assert x.dual == pytest.approx(23.097478714515493, 1e-12)

    d = Dual(2.123, 3.45)
    x = d.cosh()
    assert x.real == pytest.approx(4.237920250508109, 1e-12)
    assert x.dual == pytest.approx(14.207956211614606, 1e-12)

    d = Dual(0,1)
    x = d.cosh()
    assert x.real == pytest.approx(1, 1e-12)
    assert x.dual == pytest.approx(0, 1e-12)
    
    d = Dual(np.pi / 4, 1)
    x = d.cosh()
    assert x.real == pytest.approx(1.324609089252006, 1e-12) 
    assert x.dual == pytest.approx(0.8686709614860095, 1e-12)  

    d = Dual(0, 0)
    x = d.cosh()
    assert x.real == pytest.approx(1, 1e-12) 
    assert x.dual == pytest.approx(0, 1e-12)  


    d = Dual(-np.pi / 4, 1)
    x = d.cosh()
    assert x.real == pytest.approx(1.324609089252006, 1e-12) 
    assert x.dual == pytest.approx(-0.8686709614860095, 1e-12)  


    #now lets tets tans

    d = Dual(np.pi/4, 2)
    x = d.tanh()
    assert x.real == pytest.approx(0.6557942026326724, 1e-12)
    assert x.dual == pytest.approx(1.1398679275867547, 1e-12)

    d = Dual(2.123, 3.45)
    x = d.tanh()
    assert x.real == pytest.approx(0.9717616032972388, 1e-12)
    assert x.dual == pytest.approx(0.19209388293227045, 1e-12)


    d = Dual(0,1)
    x = d.tanh()
    assert x.real == pytest.approx(0, 1e-12)
    assert x.dual == pytest.approx(1, 1e-12)
    
    d = Dual(np.pi / 4, 1)
    x = d.tanh()
    assert x.real == pytest.approx(0.6557942026326724, 1e-12) 
    assert x.dual == pytest.approx(0.5699339637933774, 1e-12)  

    d = Dual(0, 0)
    x = d.tanh()
    assert x.real == pytest.approx(0, 1e-12) 
    assert x.dual == pytest.approx(0, 1e-12)  


    d = Dual(-np.pi / 4, 1)
    x = d.tanh()
    assert x.real == pytest.approx(-0.6557942026326724, 1e-12) 
    assert x.dual == pytest.approx(0.5699339637933774, 1e-12)  


def test_dual_reverse_division_by_zero():
    """  
    tests the handling of dividing a scaler number by a number with a zero real component
    """ 

    #initialise dual number with 0 real component
    d1 = Dual(0, 1)
    
    # test that when a acaler is divided by a dual number with zero real we get an error
    with pytest.raises(ZeroDivisionError, match="Division by a dual number with a zero real part is undefined."):
        2 / d1


def test_dual_division_by_zero():
    """  
    tests the handling of dividing a dual number by an int 0
    """

    #initialise dual number with 0 real component
    d1 = Dual(1, 1)
    
    # test that when a acaler is divided by a dual number with zero real we get an error
    with pytest.raises(ZeroDivisionError, match="Division by 0 is not defined"):
        d1 / 0

def test_negation():

    """  
    Tests that negation is properly handled for dual numbers

    """

    d1 = Dual(1,1)

    d2 = -d1

    assert d2 == Dual(-1,-1)


def test_power_raise_valid():
    """  
    Tests the type handling of the power operator
    
    """

    d1 = Dual(1,1)
    stng = "test"

    with pytest.raises(TypeError, match="can only raise Dual to Dual, int or float"):
        d1**stng

def test_sqrt():
    """   
    Tests functionality of sqrt methdod
    
    """

    # Test case: Positive real part
    d1 = Dual(4, 2)
    result = d1.sqrt()
    assert np.isclose(result.real, 2.0)
    assert np.isclose(result.dual, 0.5)

    d2 = Dual(-2,1)

    with pytest.raises(ValueError, match = "Square root is undefined for a non positive real part"):
        d2.sqrt()


def test_pow_more_exceptions():
    """
    Test more of the exceptions in the power method
    if base is 0 and power is negative we get an error in the dual 
    if base is 0 and power is between 0 and 1 then in our dual part we end up raisng to to a fractional power
    """
   
    d1 = Dual(0, 2)
    with pytest.raises(ValueError, match="cannot raise 0 to negative exponents"):
        d1 ** -1

    
    with pytest.raises(ValueError, match="cannot raise 0 to negative exponents, present in Dual component of result"):
        d1 ** 0.5
