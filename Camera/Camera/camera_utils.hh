/*! \file
 *
 * \brief Utils for cameras.
 * FIXME: Ideally this should be removed by making get_exe_dir easier to include from Camera
 */                                                                                                                    \
#pragma once

#include <string>
#include <filesystem>
#include <Windows.h>

namespace camera
{
std::string get_exe_dir();
}