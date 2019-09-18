#!/bin/sh

CONFIG_TYPE=Debug
if [ $# -ge 1 ]; then
	case $1 in
		"Release")
			CONFIG_TYPE=Release
			shift 1
			;;
		"R")
			CONFIG_TYPE=Release
			shift 1
			;;
		"Debug")
			CONFIG_TYPE=Debug
			shift 1
			;;
		"D")
			CONFIG_TYPE=Debug
			shift 1
			;;
	esac
fi

echo "cmake -B build -S . -A x64 && cmake --build build --config ${CONFIG_TYPE} $@ -- /verbosity:normal"
cmake -B build -S . -A x64 && cmake --build build --config "${CONFIG_TYPE}" $@ -- /verbosity:normal
