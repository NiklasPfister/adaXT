# distutils: define_macros=CYTHON_TRACE=1
import numpy as np
from setuptools import setup, Extension, find_packages

NAME = "adaXT"
VERSION = "0.1.0"
DESCRIPTION = "A Python package for tree-based regression and classification"
PROJECT_URLS = {
    "Documentation": "",
    "Source Code": "https://github.com/NiklasPfister/adaXT"
}

# Test dependencies
TEST_DEP = [
    "scipy",
    "flake8",
    "pytest"
]

# Such that it can be pip installed:
extras = {
    "test": TEST_DEP
}


with open("README.md", 'r') as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", 'r') as f:
    REQUIRES = f.read()

USE_CYTHON = True  # TODO: get commandline input, such that a user can choose whether to compile with cython always when installing, or just the already compiled c files

# Make all pyx files for the decision_tree
ext = '.pyx' if USE_CYTHON else ".c"
include_dir = np.get_include()

# Cythonize the decision_tree
extensions = [Extension("adaXT.decision_tree.*",
                        ["src/adaXT/decision_tree/*" + ext],
                        include_dirs=[include_dir],
                        language="c++",
                        extra_compile_args=['-O3'])]

# Cythonize the criteria functions
extensions += [Extension("adaXT.criteria.*",
                         ["src/adaXT/criteria/*" + ext],
                         include_dirs=[include_dir],
                         language="c++",
                         extra_compile_args=['-O3'])]


# If we are using cython, then compile, otherwise use the c files
if USE_CYTHON:
    from Cython.Build import cythonize
    with_debug = False
    # TODO: Annotate should be false upon release, it creates the html file,
    # where you can see what is in python
    extensions = cythonize(extensions, gdb_debug=with_debug, annotate=False)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    project_urls=PROJECT_URLS,
    install_requires=REQUIRES,
    packages=find_packages(where="src"),
    package_dir={"adaXT": "./src/adaXT"},
    ext_modules=extensions,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    tests_require=TEST_DEP,
    extras_require=extras,
)
