#!/bin/sh

CONFIG_TYPE=Debug
GENERATOR="Visual Studio 15"

function check_arg()
{
	case $1 in
		"Release"|"R")
			CONFIG_TYPE=Release
			return 1;;
		"Debug"|"D")
			CONFIG_TYPE=Debug
			return 1;;
		"Ninja"|"N")
			GENERATOR=Ninja
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

echo "[BUILD.SH] CONFIG_TYPE: ${CONFIG_TYPE}"
echo "[BUILD.SH] GENERATOR: ${GENERATOR}"

if [ "${GENERATOR}" == "Ninja" ]; then
	echo "[BUILD.SH] cmd.exe /c" 'call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE='${CONFIG_TYPE}' -DCMAKE_VERBOSE_MAKEFILE=ON && cmake --build build'
	cmd.exe /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE='${CONFIG_TYPE}' -DCMAKE_VERBOSE_MAKEFILE=ON && cmake --build build'
else
	echo "[BUILD.SH] cmake -B build -S . -A x64"
	cmake -B build -S . -A x64
	echo "[BUILD.SH] cmake --build build --config ${CONFIG_TYPE} $@ -- /verbosity:normal"
	cmake --build build --config "${CONFIG_TYPE}" $@ -- /verbosity:normal
fi
