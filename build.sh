#!/bin/sh

CONFIG_TYPE=Debug
if [ $# -ge 1 ] && [ $1 == "Release" ]; then
	CONFIG_TYPE="$1"
	shift 1
fi

cmake -B build -S . -A x64
cmake --build build --config "${CONFIG_TYPE}" $@ -- /verbosity:normal
