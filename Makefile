install:
	pip install -e .

build_ext:
	python setup.py build_ext --inplace

clean:
	rm -f src/adaXT/decision_tree/*.so src/adaXT/decision_tree/*.html src/adaXT/decision_tree/*.c src/adaXT/decision_tree/*.pyd src/adaXT/decision_tree/*.cpp
	rm -rf src/adaXT.egg-info
	rm -rf build/

lint:
	cython-lint src/*