# distutils: define_macros=CYTHON_TRACE=1
import os
import numpy as np
from setuptools import setup, find_packages, Extension


packages = find_packages(where='src')

USE_CYTHON = True #TODO: get commandline input, such that a user can choose whether to compile with cython always when installing, or just the already compiled c files

ext = '.pyx' if USE_CYTHON else ".c"
include_dirs = np.get_include()
extensions = [
    Extension("adaXT.decision_tree._func_wrapper", ["src/adaXT/decision_tree/_func_wrapper"+ext], include_dirs=[include_dirs]),
    Extension("adaXT.decision_tree._criteria", ["src/adaXT/decision_tree/criteria_cy"+ext], include_dirs=[include_dirs]),
    Extension("adaXT.decision_tree._splitter", ["src/adaXT/decision_tree/_splitter"+ext], include_dirs=[include_dirs])
    #Extension("adaXT.decision_tree._tree", ["src/adaXT/decision_tree/_tree"+ext], include_dirs=[include_dirs])
    ]

if USE_CYTHON:
    from Cython.Build import cythonize
    with_debug = False
    extensions = cythonize(extensions, gdb_debug=with_debug, annotate=True) #TODO: Annotate should be false upon release, it creates the html file, where you can see what is in python
extensions += [Extension("adaXT.decision_tree._tree", ["src/adaXT/decision_tree/_tree.py"])]

setup(
    name='adaXT',
    ext_modules=extensions
)