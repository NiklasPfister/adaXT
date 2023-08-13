import os
import numpy as np
from setuptools import setup, find_packages, Extension


packages = find_packages(where='src')

USE_CYTHON = True #TODO: get commandline input, such that a user can choose whether to compile with cython always when installing, or just the already compiled c files

ext = '.pyx' if USE_CYTHON else ".c"
include_dirs = np.get_include()
extensions = [
    Extension("adaXT.decision_tree._splitter", ["src/adaXt/decision_tree/_splitter"+ext], include_dirs=[include_dirs]),
    Extension("adaXT.decision_tree._tree", ["src/adaXt/decision_tree/_tree"+ext], include_dirs=[include_dirs])
    ]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, annotate=True) #TODO: Annotate should be false upon release, it creates the html file, where you can see what is in python

setup(
    name='adaXT',
    ext_modules=extensions,
)