import cythonbuilder
import os

files = os.listdir("./src/adaXT/decision_tree/")
files = [x for x in files if x.endswith(".pyx")]
print(files)
cythonbuilder.cy_interface(files=files)
