import os
from setuptools import setup, find_packages

packages = find_packages(where='src')

setup(
    name='adaXT',
    packages=['decision_tree'],
    install_requires=["numpy"]
)