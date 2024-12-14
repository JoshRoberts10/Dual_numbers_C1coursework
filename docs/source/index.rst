.. dual_autodiff documentation master file, created by
   sphinx-quickstart on Tue Dec  3 13:46:20 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dual_autodiff documentation
===========================

*Author*: Joshua Roberts

Welcome to the documentation for dual_autodiff package. This package handles the initilaisation and manipulation of dual numbers.

Dual numbers are numbers that contain a real element and a dual element. This system has many applictaions in automatic differentiation, geometry and mathematical computing.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   example_notebook
   comparison


Installation
------------

Ensure you have Python 3.10 or higher. Install the package in by first cloning the repository and then installing the package and its dependencies with:

.. code-block:: bash

    pip install .

From within the project folder.

Alternatively one may install the package via the python wheel files in the dist folder. 

There also exist a cython implementation of the package for increased performance. The cython impleThe cythonised version of the package is located in the dual_autodiff_x folder. 
This may be installed via running

.. code-block:: bash

    pip install . 


from inside the dual_autodiff_x folder. Alternatively dual_autodiff_x may be installed using the wheels located in the dual_autodiff_x/wheelhouse folder.
These are for the linux system which may be accessed either via the linux machine or via the docker container provided.

Usage
------------

example of how to use the package:

.. code-block:: python

    from dual_autodiff import Dual

    dual_number = Dual(1,2)



For further reading on the concept of Dual numbers please see
---------------------------------------------------------------

    - `Dual numbers Wikipedia page <https://en.wikipedia.org/wiki/Dual_number>`_
    - `Dual numbers for first order sensitivity analysis <https://ceid.utsa.edu/HYPAD/wp-content/uploads/sites/50/2023/04/3DualNumbers-12.pdf>`_
