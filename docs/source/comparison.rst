Comparison of Dual Numbers
=====================================

In this package the dunder __eq__ has been defined such that it evaluates True when two dual numbers are compared that share the same
real dual components.  Moreover it is the authors choice that a comparison between a dual number object with no 
dual component and a scalar matching the dual number's real component will  evaluate to True.

However it will be noted that some alternative dual number packages only consider the real part of a dual number during comparisons.
This stems from the perspective that epsilon is used to represent an infintesimal component and thus should be ignored in comparison operations.
It is the opinion of the Author that implimenting such functionality may lead to confusion for the end user.

As a result the only comparison operator that has been implemented is __eq__ which has logic stated above. Other comparson operators such as less than, greater
than etc... have been omiitted to eliminate the possibility of misinterpretation. Users requiring such comparisons may implement them as needed by directly accessing the real and dual components of dual number objects.