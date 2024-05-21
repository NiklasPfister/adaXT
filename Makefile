install:
	pip install -e .

build_ext:
	python setup.py build_ext --inplace

clean:
	rm -f ./src/adaXT/decision_tree/*.so ./src/adaXT/decision_tree/*.html ./src/adaXT/decision_tree/*.cpp
	rm -f ./src/adaXT/criteria/*.so ./src/adaXT/criteria/*.html ./src/adaXT/criteria/*.cpp
	rm -f ./src/adaXT/predict/*.so ./src/adaXT/predict/*.html ./src/adaXT/predict/*.cpp
	rm -f ./src/adaXT/leaf_builder/*.so ./src/adaXT/leaf_builder/*.html ./src/adaXT/leaf_builder/*.cpp

lint:
	cython-lint src/* --max-line-length=127
