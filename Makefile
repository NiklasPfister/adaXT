install:
	pip install -e .

build_ext:
	python setup.py build_ext --inplace

clean:
	rm src/adaXT/decision_tree/*.so src/adaXT/decision_tree/*.html 

clean_c: clean
	rm src/adaXT/decision_tree/*.c
