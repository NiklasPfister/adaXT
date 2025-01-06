# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
from setuptools import setup, Extension, find_packages
import os

NAME = "adaXT"
VERSION = "1.4.0"
DESCRIPTION = "A Python package for tree-based regression and classification"
PROJECT_URLS = {
    "Documentation": "https://NiklasPfister.github.io/adaXT/",
    "Source Code": "https://github.com/NiklasPfister/adaXT",
}

# Test dependencies
TEST_DEP = ["scipy", "flake8", "pytest"]

# Such that it can be pip installed:
extras = {"test": TEST_DEP}


with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", "r") as f:
    REQUIRES = f.read()

USE_CYTHON = True

DEBUG = False

PROFILE = False

# Make all pyx files for the decision_tree
ext = ".pyx" if USE_CYTHON else ".cpp"
include_dir = np.get_include()

modules = ["base_model", "utils.utils"]
modules += [
    "criteria.criteria",
    "criteria.crit_helpers",
]
modules += [
    "decision_tree._decision_tree",
    "decision_tree.decision_tree",
    "decision_tree.nodes",
    "decision_tree.splitter",
    "decision_tree.tree_utils",
]
modules += ["leaf_builder.leaf_builder"]
modules += ["predictor.predictor"]
modules += ["random_forest.random_forest"]


def get_cython_extensions() -> list[Extension]:
    source_root = os.path.abspath(os.path.dirname(__file__))
    source_root = os.path.join(source_root, "src")
    extensions = []

    for module in modules:
        module = "adaXT." + module
        module_names = module.split(".")
        source_file = os.path.join(source_root, *module_names)

        pyx_source_file = source_file + ".pyx"
        # if it does not exist as a pyx file, continue
        if not os.path.exists(pyx_source_file):
            continue

        dep_files = []
        dep_files.append(source_file + ".pxd")
        if DEBUG:
            comp_args = ["-O1"]
        else:
            comp_args = ["-O3"]
        macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        if PROFILE:
            macros.append(("CYTHON_TRACE", "1"))

        extensions.append(
            Extension(
                module,
                sources=[pyx_source_file],
                language="c++",
                depends=dep_files,
                extra_compile_args=comp_args,
                include_dirs=[include_dir],
                define_macros=macros,
            )
        )
        # XXX hack around setuptools quirk for '*.pyx' sources
        extensions[-1].sources[0] = pyx_source_file
    return extensions


def run_build():
    extensions = get_cython_extensions()
    if USE_CYTHON:
        from Cython.Build import cythonize
        from Cython.Compiler.Options import get_directive_defaults

        compiler_directives = get_directive_defaults()
        compiler_directives.update(
            {
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "initializedcheck": False,
                "nonecheck": False,
            }
        )

        if PROFILE:
            compiler_directives["profile"] = True
            compiler_directives["linetrace"] = True
            compiler_directives["binding"] = True

        extensions = cythonize(
            extensions,
            gdb_debug=False,
            annotate=True,
            language_level="3",
            compiler_directives=compiler_directives,
            verbose=True,
        )
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        project_urls=PROJECT_URLS,
        install_requires=REQUIRES,
        python_requires=">=3.10",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        include_dirs=[include_dir],
        ext_modules=extensions,
        package_data={
            "adaXT.criteria": ["*.pxd", "*.pyi", "*.py"],
            "adaXT.decision_tree": ["*.pxd", "*.pyi", "*.py"],
            "adaXT.leaf_builder": ["*.pxd", "*.pyi", "*.py"],
            "adaXT.predictor": ["*.pxd", "*.pyi", "*.py"],
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
        ],
        extras_require=extras,
        zip_safe=False,
    )


if __name__ == "__main__":
    run_build()
