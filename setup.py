# distutils: define_macros=CYTHON_TRACE=1
import numpy as np
from setuptools import setup, Extension, find_packages
import glob
import os
USE_CYTHON = True #TODO: get commandline input, such that a user can choose whether to compile with cython always when installing, or just the already compiled c files

# Make all pyx files for the decision_tree
ext = '.pyx' if USE_CYTHON else ".c"
include_dir = np.get_include()
extensions = [Extension("adaXT.decision_tree.*", ["src/adaXT/decision_tree/*" + ext], include_dirs=[include_dir])]

# If we are using cython, then compile, otherwise use the c files
if USE_CYTHON:
    from Cython.Build import cythonize
    with_debug = True
    extensions = cythonize(extensions, gdb_debug=with_debug, annotate=False)

setup(
    name='adaXT',
    packages=find_packages(),
    ext_modules=extensions
)