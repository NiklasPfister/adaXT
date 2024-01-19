install:
	pip install -e .

build_ext:
	python setup.py build_ext --inplace

clean:
	rm src/adaXT/decision_tree/*.so src/adaXT/decision_tree/*.html src/adaXT/decision_tree/*.cpp
	rm src/adaXT/crteria/*.so src/adaXT/crteria/*.html src/adaXT/crteria/*.cpp src/adaXT/utils/*.so
lint:
	cython-lint src/* --max-line-length=127