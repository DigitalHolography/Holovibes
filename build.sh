#!/bin/sh

CONFIG_TYPE=Debug
GENERATOR="Visual Studio 15"

function check_arg()
{
	case "$1" in
		"Release"|"R"|"r")
			CONFIG_TYPE=Release
			return 1;;
		"Debug"|"D"|"d")
			CONFIG_TYPE=Debug
			return 1;;
		"Ninja"|"N"|"n")
			GENERATOR=Ninja
			return 1;;
		"NMake"|"NM"|"nm")
			GENERATOR="NMake Makefiles"
			return 1;;
		"Visual Studio"*)
			GENERATOR="$1"
			return 1;;
		*)
			return 0;;
	esac
}

while [ $# -ge 1 ]; do
	check_arg "$1"
	if [ $? -eq 1 ]; then
		shift 1
	else
		break
	fi
done

echo "[BUILD.SH] CONFIG_TYPE: ${CONFIG_TYPE}"
echo "[BUILD.SH] GENERATOR: ${GENERATOR}"

if [ -z "${GENERATOR##Visual Studio*}" ]; then
	echo "[BUILD.SH] cmake -G ${GENERATOR} -B build -S . -A x64"
	cmake -G "${GENERATOR}" -B build -S . -A x64
	echo "[BUILD.SH] cmake --build build --config ${CONFIG_TYPE} $@ -- /verbosity:normal"
	cmake --build build --config "${CONFIG_TYPE}" $@ -- /verbosity:normal
else
	echo "[BUILD.SH] cmd.exe /c" 'call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake -B build -S . -G "'"${GENERATOR}"'" -DCMAKE_BUILD_TYPE='${CONFIG_TYPE}' -DCMAKE_VERBOSE_MAKEFILE=ON && cmake --build build '"$@"''
	cmd.exe /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake -B build -S . -G "'"${GENERATOR}"'" -DCMAKE_BUILD_TYPE='${CONFIG_TYPE}' -DCMAKE_VERBOSE_MAKEFILE=ON && cmake --build build '"$@"''
fi
