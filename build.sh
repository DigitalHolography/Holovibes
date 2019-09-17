#!/bin/sh

CONFIG_TYPE=Debug
if [ $# -ge 1 ] && [ $1 == "Release" ]; then
	CONFIG_TYPE="$1"
	shift 1
fi

echo "cmake -B build -S . -A x64"
cmake -B build -S . -A x64

echo "cmake --build build --config ${CONFIG_TYPE} $@ -- /verbosity:normal"
cmake --build build --config "${CONFIG_TYPE}" $@ -- /verbosity:normal
