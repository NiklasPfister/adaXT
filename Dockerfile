ARG MANYLINUX=manylinux_2_28_x86_64
FROM quay.io/pypa/${MANYLINUX}
COPY . /src/
WORKDIR /src/
RUN ./build-wheels.sh
