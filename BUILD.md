# Creating a new release:

This document is intented to act as a reference on how to build future releases.

## Steps
* Make sure docker is installed and functional
* Change directory to the root directory of adaXT
* run ''' docker build --tag adaxt . ''' - This will take a while as it is creating the wheels for many different linux versions.
* Then run ''' docker cp adaxt:/src/dist . ''' this will copy the built distribution to a new folder called dist in the current working directory.
* Make sure twine is installed with '''python -m pip install twine''' and set it up according to documentations with you username and password for pypi.
* Finally run '''python3 -m twine upload --repository pypi dist/*'''
