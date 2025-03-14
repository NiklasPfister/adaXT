install:
	pip install -e .

build_ext:
	python setup.py build_ext --inplace

clean:
	find ./src | grep -i .so | xargs rm -rf
	find ./src | grep -i .cpp | xargs rm -rf
	find ./src | grep -i .html | xargs rm -rf
	find ./src | grep -i egg-info | xargs rm -rf
	find ./src | grep -i pycache | xargs rm -rf

lint:
	cython-lint src/* --max-line-length=127

mkdocs_install:
	pip install mkdocs mkdocs-material mkdocstrings 'mkdocstrings[python, cython]' mkdocs-autorefs pymdown-extensions

mkdocs: mkdocs_install
	mkdocs serve


test_pypi:
	pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple adaXT
