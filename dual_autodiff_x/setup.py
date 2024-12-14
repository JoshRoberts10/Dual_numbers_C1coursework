from setuptools import setup, Extension
from Cython.Build import cythonize

#setup fiel addapted from C1 Lecture notes


# Define the extensions (Cython modules)
extensions = [
    Extension("dual_autodiff_x.dual", ["dual.pyx"]),
]


VERSION = "1.1.1"


# Call setup with cythonized extensions
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
    package_dir={"dual_autodiff_x": "."},  # Map the package to the current directory
    packages=["dual_autodiff_x"],
    package_data={"dual_autodiff_x": ["*.pyd", "*.so"]},  # Include compiled files
    exclude_package_data={"dual_autodiff_x": ["*.pyx", "*.py"]},  # Exclude source files
    zip_safe=False,
)