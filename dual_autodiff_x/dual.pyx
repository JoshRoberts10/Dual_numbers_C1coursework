import numpy as np
import math


class Dual:
    """
    A class used to represent a Dual number, which has the form 
    
    .. math:: 
        a + b \epsilon, 

    where :math:`a` is the real component and :math:`b` is the dual component.

    :math:`\epsilon` satisfies the property
     
    .. math:: 
        \epsilon^2 = 0.
    
    
    Dual numbers have particular applications in automatic differentiation, geometry and mathematical computing. For referances for any
    formulas in this package please see references below.

   
    Attributes
    ----------- 
    real : int, float
        The real part of the dual number
    dual : int, float
        The dual part of the dual number


        
    references
    ----------- 
    - `Dual numbers Wikipedia page <https://en.wikipedia.org/wiki/Dual_number>`_
    - `Dual numbers for first order sensitivity analysis <https://ceid.utsa.edu/HYPAD/wp-content/uploads/sites/50/2023/04/3DualNumbers-12.pdf>`_
    """


    def __init__(self, real, dual):
        """
        Initailises the Dual object

        
        Parameters
        -----------
        real : float
            The real part of the dual number
        dual : float
            The dual part of the dual number

        Raises
        ------
        TypeError
            If the `real` or `dual` component is not a float or integer.
        ValueError
            If the `real` or `dual` component is NaN (not a number).
        ValueError
            If the `real` or `dual` component is infinite.

        Notes
        ----- 
        The real and dual components must be finite and cannot be NaN or infinity, this is to ensure correct behavior in numerical
        calculation.

        Examples
        -------- 
        >>> d = Dual(2.0, 1.0)
        >>> print(d)
        Dual(2.0, 1.0)
        """

        # checks that the inputs in intialisation are valid 
        if not isinstance(real, (int, float)):
            raise TypeError("real component must be either a float or an integer")
        if not isinstance(dual, (int, float)):
            raise TypeError("dual component must be either a float or an integer")
        

        # Check that real and dual components are not nan, as these are technically floats
        if np.isnan(real):
            raise ValueError("real component cannot be nan")
        if np.isnan(dual):
            raise ValueError("dual component cannot be nan")
        
        # check that real and dual components are not infinite, this is to ensure no ambiguity as for some functions infinity has
        # undefined action
        if np.isinf(real):
            raise ValueError("real component cannot be inf")
        if np.isinf(dual):
            raise ValueError("dual component cannot be inf")


        self.real = real
        self.dual = dual




    def __str__(self):
        """
        Returns Dual object in string format for readability
        The string is formatted as Dual(real = x, dual = y)

        
        Returns
        --------
        Str
            String representation of the dual number

            
        Examples
        -------- 
        >>> d = Dual(2,3)
        >>> print(d)
        Dual(real = 2, dual = 3)
        """

    
        return "Dual(real = {}, dual = {})".format(self.real, self.dual)
    



    def add(self,other):

        return Dual(self.real+other.real, self.dual + other.dual)









    def __add__(self, other):
        """
        Adds two dual numbers or a dual number and a scalar

        
        Parameters
        -----------
        other : Dual, int, float
            other (dual) number to be added

            
        Returns
        -------
        Dual
            The addition of the two (dual) numbers

            
        Raises
        ------
        TypeError
            If other is not a Dual, int, or float.
        

        Examples
        -------- 
        >>> x = Dual(2, 3)
        >>> y = Dual(1, 4)
        >>> x + y
        Dual(3, 7)

        >>> x = Dual(2, 3)
        >>> x + 1
        Dual(3, 3)
        """

        # check if we are adding a dual number to the current dual number
        if isinstance(other, Dual):
            # calculates real part of added dual number
            new_real = self.real + other.real
            # calculates dual part of added dual number
            new_dual = self.dual + other.dual
            return Dual(new_real, new_dual)
        
        # if we are adding a scaler to the current dual, then it only adds to the real part
        elif isinstance(other, (int, float)):
            new_real = self.real + other
            return(Dual(new_real, self.dual))
        
        else:
            raise TypeError("Unsupported type for addition {}".format(type(other)))
        


    def __radd__(self, other):
        """
        Handles addition when the dual number is on the right side of the addition.
        This is used when adding a scalar on the left to a dual object.
        

        Parameters
        ----------
        other : int, float
            The scalar value for which we add our dual

            
        Returns
        -------
        Dual
            The result of the scalar plus dual
     
            
        Raises
        ------
        TypeError
            If other is not an int or float.

            
        Examples
        --------
        >>> d = Dual(2, 3)
        >>> 5 + d
        Dual(7, 3)
        """

        # checks if the object we are subtracting from is a scaler
        if isinstance(other, (int, float)):
            return Dual(other + self.real, self.dual)

        else:
            raise TypeError("Unsupported type for addition {}".format(type(other)))


    

    def __sub__(self, other):
        """
        Subtracts a dual number or scalar from the current dual number instance

        
        Parameters
        ----------
        other : Dual, int or float
            The (Dual) number to subtract

            
        Returns
        -------
        Dual
            The result of the subtraction

        Raises
        ------
        TypeError
            If other is not a Dual, int, or float.

            
        Examples
        --------
        >>> x = Dual(2, 3)
        >>> y = Dual(1, 4)
        >>> x - y
        Dual(1, -1)

        >>> x = Dual(2, 3)
        >>> x - 1
        Dual(1, 3)
        """

        # check if we are subtracting a dual number to the current dual number
        if isinstance(other, Dual):
            new_real = self.real - other.real
            new_dual = self.dual - other.dual
            return Dual(new_real, new_dual)
        
        # if we are minusing a scaler to the current dual
        elif isinstance(other, (int, float)):
            new_real = self.real - other
            return Dual(new_real, self.dual)
        
        else:
            raise TypeError("Unsupported type for subtraction {}".format(type(other)))
        

    def __rsub__(self, other):
        """
        Handles subtraction when Dual object is on the right of the subtraction.
        This is used when subtracting a Dual object from a scalar


        Parameters
        ----------
        other : int or float
            The scaler value from which our dual number is subtracted

            
        Returns
        -------
        Dual
            The result of the subtraction of scalar - self.

            
        Raises
        ------
        TypeError
            If other is not an int or float.
            
        Examples
        --------
        >>> x = Dual(1, 4)
        >>> 2 - x
        Dual(1, -4)
        """

        # checks if the object we are subtracting from is a scaler
        if isinstance(other, (int, float)):
            return Dual(other-self.real, -self.dual)

        else:
            raise TypeError("Unsupported type for subtraction {}".format(type(other)))
    

    def __mul__(self, other):
        """
        Multiplies one dual number by another Dual number or Scalar. 
        
        This follows 
        
        .. math::
            (a + bε)(c + dε) = ac +ad\epsilon + bc\epsilon 

        Parameters
        ----------
        other : Dual or int or float
            The Dual number or scalar to multiply by

            
        Returns
        -------
        Dual
            The result of multiplication.

            
        Raises
        ------
        TypeError
            If other is not a Dual, int, or float.   
            
        Examples
        --------
        >>> x = Dual(2, 3)
        >>> y = Dual(1, 4)
        >>> x * y
        Dual(2, 11)

        >>> x = Dual(2, 3)
        >>> x * 2
        Dual(4, 6)
        """

        # checks if we are multiplying by a dual number
        if isinstance(other, Dual):
            new_real = self.real * other.real
            new_dual = (self.real * other.dual) + (self.dual * other.real)
            return Dual(new_real, new_dual) 
        
        # checks if multiplying by scaler
        elif isinstance(other, (int, float)):
            return Dual(self.real*other, self.dual*other)
        
        else:
            raise TypeError("Unsupported type for multiplication {}".format(type(other)))
        

    def __rmul__(self,other):
        """
        Handles multiplication when Dual object is on the right and scalar object on the left

        This follows 
        
        .. math::
            (a)(c + dε) = ac +ad\epsilon 

        
        Parameters
        ----------
        other : int or float
            The scalar number on the left side of our multiplication with dual

            
        Returns
        -------
        Dual
            The result of multiplying the dual and scalar.


        Raises
        ------
        TypeError
            If other is not an int or float.

            
        Examples
        --------
        >>> d = Dual(2, 3)
        >>> 2 * d
        Dual(4, 6)
        """

        # check that other is a scalar or int value
        if isinstance(other, (int, float)):
            return(Dual(self.real*other, self.dual*other))
                   
        else:
            raise TypeError("Unsupported type for multiplication {}".format(type(other)))
                   

    
        
    def __truediv__(self, other):
        """
        Divides one dual number by a dual number or scalar.

        For two dual numbers, division is defined as:

        .. math::

            \\frac{a + b \\epsilon}{c + d \\epsilon} = \\frac{a}{c} + \\frac{bc - ad}{c^2} \\epsilon, \\quad \\text{for } c \\neq 0

        When dividing by a scalar, this is equivalent to dividing by a dual
        number with a zero dual component :math:`c + 0 \\epsilon`.

        Parameters
        ----------
        other : Dual, int, float
            The Dual number or scalar to divide by
        
        Returns
        -------
        Dual
            The result of the division.

        Raises
        ------
        ZeroDivisionError
            If the divisor is zero or the real part of a dual divisor is zero.

        TypeError
            If other is not a Dual, int or float.
        
        Examples
        --------
        >>> x = Dual(2, 3)
        >>> y = Dual(1, 4)
        >>> x / y
        Dual(2.0, -5.0)

        >>> x = Dual(2, 3)
        >>> x / 2
        Dual(1.0, 1.5)

        """

        # checks if the divisor is also a Dual number
        if isinstance(other, Dual):

            #checks if the real component of divisor is zero.
            if other.real ==0:
                raise ZeroDivisionError("The real part of the divisor is 0, division is not defined")

            new_real = self.real/other.real
            new_dual = ((self.dual * other.real - self.real*other.dual)/other.real**2)
            return Dual(new_real, new_dual)

        # checks if the division is a scalar
        elif isinstance(other, (int, float)):
            if other==0:
                 raise ZeroDivisionError("Division by 0 is not defined")

            return Dual(self.real / other, self.dual / other)
        
        else:
            raise TypeError("Unsupported type for division {}".format(type(other)))


    def __rtruediv__(self,other):
        """
        Handles division by a dual number when the numerator is an integer or float.

        Division of a real number by a dual number is defined as:

        .. math::

            \\frac{a}{c + d \\epsilon} = \\frac{a}{c} + \\frac{-ad}{c^2} \\epsilon, \\quad \\text{for } c \\neq 0

        
        Parameters
        ----------
        other : scaler or int
            The scaler numerator
        
        Returns
        -------
        Dual
            The result of dividing other by self.

            
        Raises
        ------
        ZeroDivisionError
            If the real part of the Dual denominator is zero.

        TypeError
            If other is not an int or float.

        Examples
        --------
        >>> d = Dual(2, 3)
        >>> 4 / d
        Dual(2.0, -3.0)

        """

        # checks if real part of dual in denominator is zero, if so then the division is not defined
        if self.real == 0:
            raise ZeroDivisionError("Division by a dual number with a zero real part is undefined.")

        # checks if the numerator is a scaler value such that divison is defined
        elif isinstance(other, (int, float)):
            new_real = other / self.real
            new_dual = (-other * self.dual / self.real**2)
            return (Dual(new_real, new_dual))
        
        else:
            raise TypeError("Unsupported type for division {}".format(type(other)))



    def __neg__(self):
        """
        Returns a negative version of the dual number
        

        Returns
        -------
        Dual
            The negative version of the Dual number

            
        Examples
        --------
        >>> d = Dual(2, 3)
        >>> -d
        Dual(-2, -3)
        """

        return Dual(-self.real, -self.dual)
    

    def __pow__(self, power):
        """
        Raises the Dual number instance to a Dual number or scalar

        The formula for raising a Dual number :math:`(a + b\epsilon)` to the power of another Dual number :math:`(c + d\epsilon)` is defined as:

        .. math::

            (a+b\epsilon)^{c+d\epsilon} = a^{c} + a^{c-1} (ad\ln(a) +cb)\epsilon

            
        When raising a Dual number to a scalar, this is a special case where the power has a zero 
        dual component. The formula becomes:

        .. math::

            (a+b\epsilon)^{c} = a^{c} + (a^{c-1} cb)\epsilon

        Parameters
        ----------
        power : Dual or int or float
            The exponent to raise the dual number to

        Returns
        -------
        Dual
            The result of raising the Dual number to the given power.


        Raises
        ------
        ValueError
            If operation is not defined for given values.

        TypeError
            If power is not a Dual, int, or float.

        """


        #checks our power is a dual, int or float
        if not isinstance(power, (Dual, int, float)):
            raise TypeError("can only raise Dual to Dual, int or float")

        # if our power is a scalar we convert it to a dual object, with no dual component
        if isinstance(power, (int,float)):
            power = Dual(power, 0)


        
        # checks if our power has a 0 dual component, meaning we are raising to a scaler and may use formula 2
        # the reason we have converted scalers to a dual and then are checking for no dual component is to ensure
        # code handles the case where i input a scaler object vs inputting a dual object with no dual component in the
        # same way 

        #considering scalers
        if power.dual==0:

            # lets consider some edge cases
            # there is a more compact way to write these edge cases, however I believe for better readibility its easier if i break
            # them down as to be more explicit

            # 1) when real base is 0 raised to 0, we get a 0^0 error
            if self.real == 0 and power.real ==0:
                raise ValueError("0^0 is not defined")
            
            # 2) if a<0 and our power is fraction we get an error
            if self.real<0 and type(power.real) is not int:
                raise ValueError("cannot raise negative numbers to fractional powers")
            
            # 3) if real base is 0 and scaler power is 1, we end up wuth 0^0 in dual component 
            if self.real==0 and power.real ==1:
                raise ValueError("0^0 is not defined and is present in dual component")
            
            # 4) if base is 0 and power is negative we get an error in the dual 
            if self.real ==0 and power.real<0:
                raise ValueError("cannot raise 0 to negative exponents")

            # 5) if base is 0 and power is between 0 and 1 then in our dual part we end up raisng to to a fractional power
            if self.real ==0 and 0<power.real<1:
                raise ValueError("cannot raise 0 to negative exponents, present in Dual component of result") 
            
            # if no edge cases then may use general form 
            new_real = self.real ** power.real
            new_dual = power.real * self.dual * (self.real ** (power.real - 1))
            return Dual(new_real, new_dual)

        # now we are considering the cases when we are raising a dual to a dual
        if power.dual!= 0:


            # lets handle some edge cases

            #1) if my real base is negative or zero and i have a dual component (handled by the above if) then i have a negative log or zero log
            if self.real<=0:
                raise ValueError("Cannot raise negtive or 0 real dual to a dual with non zero dual component")
            

            # edge cases handled use general formuala 
            new_real = self.real ** power.real
            new_dual = (self.real ** power.real) * (power.dual * np.log(self.real) + (power.real * self.dual) / self.real)

            return Dual(new_real, new_dual)
        



    def __rpow__(self, other):
        """
        Method used when raising a scalar to a dual power.

        The formula for raising a scalar :math:`a` to the power of a Dual number 
        :math:`(c + d\epsilon)` is defined as:

        .. math::
            a^{c + d\epsilon} = a^c + (a^c \ln(a)d) \epsilon


        Parameters
        ----------
        other : int or float
            The Base of the exponent

        Returns
        -------
        Dual
            The result of raising the scalar number to the dual power.


        Raises
        ------
        ValueError
            If operation is not defined for given values.

        TypeError
            If power is not a Dual, int, or float.

        """

        # to ensure correctness and consistancy with other method, i may turn our scalar base into a dual number with a zero dual
        #componet
        base = Dual(other, 0)

        return base**self



    def __eq__(self, other):
        """
        Method for evaluating if two dual numbers are equal

        There is some ambiguity when comparing Dual numbers. In this package 2 dual numbers are defined to be equal if both 
        their real and Dual parts are equal. Moreover a dual number with a zero dual component is considered equal to a scalar
        if their real components match.

        
        Parameters
        ----------
        other : int or float or dual
            The obejct to be compared to 

        Returns
        -------
        Bool
            whether the 2 objects are equal

        Raises
        ------
        TypeError
            If comparing to a non int, float or Dual type

        Notes
        -----
        - To account for floating-point precision issues, equality is checked using a tolerance.


        Examples
        --------
        >>> d1 = Dual(3.0, 0.0)
        >>> d2 = Dual(3.0, 2)
        >>> d1 == d2
        False 

        >>> d3 = Dual(3.0, 0.0)
        >>> d3 == 3.0
        True  
        
        """
        
        # if comparing to a scaler, converts scaler to dual number with a zero dual component
        if isinstance(other, (int, float)):
            other = Dual(other, 0)

        # As often in Dual numbers one is using to calulate derivitives, numbers often reach max digits to be represented in floating 
        # and thus succumb to floating point precision, the equality checks if the 2 numbers are close with some tolerence
        if isinstance(other, Dual):
            return math.isclose(self.real, other.real, rel_tol = 1e-12) and math.isclose(self.dual, other.dual, rel_tol = 1e-12)
        else:
            raise TypeError("invalid object for comparison {}".format(type(other)))
        

        
    def __req__(self, other):
        """
        Method for evaluating 2 objects, this method is called if the  dual number is on the right of the equality

        There is some ambiguity when comparing Dual numbers. In this package 2 dual numbers are defined to be equal if both 
        their real and Dual parts are equal. Moreover a dual number with a zero dual component is considered equal to a scalar
        if their real components match.

        
        Parameters
        ----------
        other : int or float
            The obejct to be compared to 

        Returns
        -------
        bool
            whether the 2 objects are equal


        Raises
        ------
        TypeError
            If comparing to a non int, float or Dual type

        Notes
        -----
        - To account for floating-point precision issues, equality is checked using a tolerance.


        Examples
        --------
  
        >>> d1 = Dual(3.0, 0.0)
        >>> x = 3
        >>> x == d1
        True  
        
        """
        
        # if comparing to a scaler, converts scaler to dual number with a zero dual component and then calls __eq__ 
        if isinstance(other, (int, float)):
            other = Dual(other, 0)
            return other == self
        else:
            raise TypeError("invalid object for comparison {}".format(type(other)))






            
    def sin(self):
        """
        Computes the sine of a dual number.

        For dual numbers, the sine function is defined as:

        .. math::
            \sin(a + b\epsilon) = \sin(a) + b \cos(a)\epsilon

        Returns
        -------
        Dual
            The result sine of the dual number.


        Examples
        --------
        >>> d = Dual(np.pi, 2)
        >>> d.sin()
        Dual(0, -2)
            
        """

        new_real = np.sin(self.real) 
        new_dual = self.dual * np.cos(self.real)

        return Dual(new_real, new_dual)
    
    def cos(self):
        """
        Computes the cosine of a dual number.

        For dual numbers, the cosine function is defined as:

        .. math::
            \cos(a + b\epsilon) = \cos(a) - b \sin(a)\epsilon 


        Returns
        -------
        Dual
            The cosine of the dual number.

        Examples
        --------
        >>> d = Dual(np.pi, 1)
        >>> d.cos()
        Dual(-1, 0)
            
        """

        new_real = np.cos(self.real) 
        new_dual = -self.dual * np.sin(self.real)

        return Dual(new_real, new_dual)
    
    def tan(self):
        """
        Computes the tangent of a dual number.

        For dual numbers, the tangent function is defined as:


        .. math::
            \\tan(a + b\epsilon) = \\frac{\sin(a + b\epsilon)}{\cos(a + b\epsilon)}

        or also written as 

        .. math::
            \\tan(a + b\epsilon) = \\tan(x) +b\\sec^{2}\epsilon


        The tangent function is undefined when :math:`a = \\frac{\pi}{2} + n\pi` for any integer :math:`n`, 
        due to division by zero in the cosine function.

        Returns
        -------
        Dual
            The tangent of the dual number.


        Raises
        ------
        ZeroDivisionError
            If the cosine of the real part is zero, making the tangent undefined.


        Examples
        --------
        >>> d = Dual(np.pi / 4, 2)
        >>> d.tan()
        Dual(1.0, 4.0)
        
        """

        # Use np.isclose to check if our cos of real part is zero due to floating point precision
        if np.isclose(np.cos(self.real), 0):
            raise ZeroDivisionError("tangent is non-defined when real component = pi/2 + n*pi")

        else:
            return self.sin()/self.cos()
    

    def sinh(self):
        """
        Computes the hyperbolic sine of a dual number.

            For dual numbers, the hyperbolic sine is defined as:

        .. math::
            \sinh(a + b\epsilon) = \sinh(a) + b \cosh(a) \epsilon


        Returns
        -------
        Dual
            The hyperbolic sine of the dual number.

        Examples
        --------
        >>> d = Dual(0, 1)
        >>> d.sinh()
        Dual(0.0, 1.0)

        """

        new_real = np.sinh(self.real)
        new_dual = self.dual * np.cosh(self.real)

        return Dual(new_real, new_dual)
    
   
    def cosh(self):
        """
        Computes the hyperbolic cosine of a dual number.

        For dual numbers, the hyperbolic cosine is defined as:

        .. math::
            \cosh(a + b\epsilon) = \cosh(a) + b \sinh(a) \epsilon

        Returns
        -------
        Dual
            The hyperbolic cosine of the dual number.
        
            
        >>> d = Dual(0.0, 1.0)
        >>> result = d.cosh()
        Dual(1.0, 0.0)
        """

        new_real = np.cosh(self.real)
        new_dual = self.dual * np.sinh(self.real)

        return Dual(new_real, new_dual)
    
    def tanh(self):
        """
        Computes the hyperbolic tangent of a dual number.

        For dual numbers, the hyperbolic tangent is defined as:

        .. math::
            \\tanh(a + b\epsilon) = \\frac{\sinh(a + b\epsilon)}{\cosh(a + b\epsilon)}

        or also as 
            
        .. math::
            \\tan(a + b\epsilon) = \\tan(x) +b sech^{2}\epsilon


        Returns
        -------
        Dual
            The hyperbolic tangent of the dual number.

        >>> d = Dual(0.0, 1.0)
        >>> result = d.tanh()
        Dual(0.0, 1.0)
        """
    
        return self.sinh()/self.cosh()
    

    def sqrt(self):
        """
        Computes the square root of a dual number.

        The square root of a dual number is defined as:

        .. math::
            \sqrt{a + b\epsilon} = \sqrt{a} + \\frac{b}{2\sqrt{a}} \epsilon

        
        Returns
        -------
        Dual
            The sqaure root of the dual number.

        Raises
        ------
        ValueError
            If the real part of the Dual number is negative.

            
        Examples
        --------
        >>> d = Dual(4, 2)
        >>> d.sqrt()
        Dual(2.0, 0.25)
        """
        # checks if real is less than 0 in which case the square root is not defined
        if self.real <=0:
            raise ValueError("Square root is undefined for a non positive real part")
            
        
        #calculates the real and dual part of the squre root
        else:
            new_real = np.sqrt(self.real)
            new_dual = self.dual/(2*np.sqrt(self.real))
            return Dual(new_real, new_dual)


    def exp(self):
        """
        Implements the exponential of a Dual number

        the exponential of a dual number is defined as

        .. math::
            \exp(a + b\epsilon) = e^{a} + b e^{a} \epsilon

        Returns
        -------
        Dual
            The exponential of the dual number.

        Examples
        --------
        >>> d = Dual(0, 3)
        >>> d.exp()
        Dual(1.0, 3.0)
        """

        new_real = np.exp(self.real)
        new_dual = self.dual * np.exp(self.real)

        return Dual(new_real, new_dual)


    def log(self):
        """
        Implements the natural logarithm of a Dual number

        The natural logarithm of a Dual number is defined by 
            
        .. math::
            \log(a + b\epsilon) = \log(a) + \\frac{b}{a}\epsilon

        Returns
        -------
        Dual
            The natural logarithm of the dual number.


        Raises
        ------
        ValueError
            If the real part of the Dual number is non-positive.

        
        Examples
        --------
        >>> d = Dual(1, 2)
        >>> d.log()
        Dual(0.0, 2.0)
        """
        
        #checks real part, if less than 0 then logarithm undefined
        if self.real<= 0:
            raise ValueError("Natural Logarithm is not defined for non-positive real parts")
        
        new_real = np.log(self.real)
        new_dual = self.dual/self.real

        return Dual(new_real, new_dual)
        
