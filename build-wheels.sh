#!/bin/bash
PLAT=manylinux_2_28_x86_64
function repair_wheel {
	wheel="$1"
	if ! auditwheel show "$wheel"; then
		echo "Skipping non-platform wheel $wheel"
	else
		auditwheel repair "$wheel" --plat "$PLAT" -w wheelhouse/
	fi
}
yum install -y libjpeg-turbo-devel

# Compile wheels
for PYBIN in /opt/python/*/bin; do
	"${PYBIN}/pip" install -r requirements.txt
	"${PYBIN}/pip" install .
	"${PYBIN}/python" setup.py sdist bdist_wheel
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
	repair_wheel "$whl"
done

rm -rf ./dist/*.whl
mv ./wheelhouse/* ./dist/
