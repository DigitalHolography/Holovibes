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

echo "build/$CONFIG_TYPE && ./Holovibes.exe"
cd build/$CONFIG_TYPE && ./Holovibes.exe