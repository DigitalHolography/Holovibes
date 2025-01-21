/*! \file
 *
 * \brief List of precompiled headers
 *
 * Precompiled header. Put here all the external includes that aren't used in a
 * cuda file to avoid recompiling it each time.
 */

#pragma once
// First, sort all the line
// To remove duplicated line, replace  ^(.*)(\r?\n\1)+$  by $1

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

// Standard Library

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <unordered_map>
#include <vector>

// C include
#include <stdint.h>
#include <sys/stat.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>

// Windows Kit
#include <Windows.h>
#include <direct.h>

// CUDA
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>

// Logger spdlog
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
