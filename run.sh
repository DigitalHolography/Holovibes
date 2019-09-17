#!/bin/sh

CONFIG_TYPE=Debug
if [ $# -ge 1 ] && [ $1 == "Release" ]; then
	CONFIG_TYPE="$1"
fi

echo "build/$CONFIG_TYPE && ./Holovibes.exe"
cd build/$CONFIG_TYPE && ./Holovibes.exe