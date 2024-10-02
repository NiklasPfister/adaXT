# Installation

We recommend installing adaXT via [pypi](https://pypi.org/project/adaXT/) using
pip. This can be done using the following command:

```bash
pip install adaXT
```

Alternatively, it can also be installed directly from the github repository:

```bash
pip install git+https://github.com/NiklasPfister/adaXT.git#egg=adaXT
```

Some new features might not yet be merged onto the main branch. If you are
feeling experimental and want to try out the current development version you can
install it with the following command:

```bash
pip install git+https://github.com/NiklasPfister/adaXT.git@Development#egg=adaXT
```

## Modifying the project and building it locally

Simple extensions such as adding a custom criteria or predict class can be
easily done without any modifications to the base package, as described
[here](/docs/user_guide/creatingCriteria.md) and
[here](/docs/user_guide/creatingCriteria.md). However, more involved changes may
require changing some of the inner workings of the package. As it is one of the
main goals of adaXT to provide an adaptable and extendable package, we have
tried to make such changes as easy as possible by keeping the code as simple as
possible.

If you want to modify the package itself, you can follow the following steps to
download the project and then build it locally.

1. **Download source code**: Either you directly download the repository github
   or you
   [create a fork on github](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).
2. **Modify code**: Modify or extend the code as you please.
3. **Build and install package**: From the project root directory you can then
   build and install the package with the command:
   ```bash
   pip install .
   ```
   This will require the
   [setuptools](https://setuptools.pypa.io/en/latest/index.html) package to be
   installed. Note that if you added new files or directories you will also need
   to modify the `setup.py` file accordingly.
4. **Use package**: Once the package is installed you can use it in the same way
   in which you used the original package.

   You can also consider creating a pull request, if you think your improvement
   or extension could be of interest to others.
