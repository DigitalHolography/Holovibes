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

// Boost
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/config.hpp>
#include <boost/program_options/environment_iterator.hpp>
#include <boost/program_options/eof_iterator.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/version.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/tokenizer.hpp>

// CUDA
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

// Logger spdlog
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
