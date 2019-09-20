#!/bin/sh

CONFIG_TYPE=Debug

function check_arg()
{
	case $1 in
		"Release"|"R")
			CONFIG_TYPE=Release
			return 1;;
		"Debug"|"D")
			CONFIG_TYPE=Debug
			return 1;;
		*)
			return 0;;
	esac
}

while [ $# -ge 1 ]; do
	check_arg $1
	if [ $? -eq 1 ]; then
		shift 1
	else
		break
	fi
done

echo "[RUN.SH] CONFIG_TYPE: ${CONFIG_TYPE}"

echo "[RUN.SH] cd build/$CONFIG_TYPE"
cd build/$CONFIG_TYPE

echo "[RUN.SH] ./Holovibes.exe"
./Holovibes.exe
