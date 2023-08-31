install:
	python -m pip install -e .

clean:
	rm src/adaXT/decision_tree/*.so src/adaXT/decision_tree/*.html 

clean_c: clean
	rm src/adaXT/decision_tree/*.c
