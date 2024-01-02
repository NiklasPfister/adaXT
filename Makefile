install:
	pip install -e .

build_ext:
	python setup.py build_ext --inplace

clean:
	rm src/adaXT/decision_tree/*.so src/adaXT/decision_tree/*.html src/adaXT/decision_tree/*.c src/adaXT/decision_tree/*.pyd
	rm -r src/adaXT.egg-info
	rm -r build/

lint:
	cython-lint src/*