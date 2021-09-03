#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Compile wheels
"${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
"${PYBIN}/pip" install inkid --no-index -f /io/wheelhouse
"${PYBIN}/python" -m unittest