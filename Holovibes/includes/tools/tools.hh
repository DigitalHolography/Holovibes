/*! \file tools.hh
 *
 * \brief Generic, widely usable functions.
 */
#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <string>
#include <ctime>
#include <cassert>

#ifndef __NVCC__
#include <qrect.h>
#else
class QString;
using uchar = unsigned char;
using uint = unsigned int;
using ushort = unsigned short;
using ulong = unsigned long;
#endif

#include <filesystem>
#include <fstream>

#include "rect.hh"
#include "hardware_limits.hh"
#include "frame_desc.hh"
#include "cufft.h"

#include "logger.hh"
#include "notifier.hh"
#include "chrono.hh"

#include <nlohmann/json.hpp>

using json = ::nlohmann::json;

std::string engineering_notation(double n, int nb_significand_digit);

/*! \brief Given a problem of *size* elements, compute the lowest number of blocks needed to fill a compute grid.
 *
 * \param nb_threads Number of threads per block.
 */
inline unsigned map_blocks_to_problem(const size_t problem_size, const unsigned nb_threads)
{
    unsigned nb_blocks =
        static_cast<unsigned>(std::ceil(static_cast<float>(problem_size) / static_cast<float>(nb_threads)));

    CHECK(nb_blocks <= get_max_blocks(), "Too many blocks required.");

    return nb_blocks;
}

template <typename T>
bool is_between(T val, T min, T max)
{
    return min <= val && val <= max;
}

namespace holovibes
{

/*! \brief return width and height with the same ratio and the max of the two being window_size */
void get_good_size(ushort& width, ushort& height, ushort window_size);

/*! \brief Preprend a string to a file path and append a number if the file already exists
 *
 * \param[in] file_path The file path to modify
 * \param[in] prepend The string to prepend
 *
 * \return The new file path
 */
std::string get_record_filename(std::string file_path, std::string prepend = Chrono::get_current_date());

/*! \brief Returns the absolute path from a relative path (prepend by the execution directory) for qt */
QString create_absolute_qt_path(const std::string& relative_path);

/*! \brief Returns the absolute path to the user Documents folder */
std::filesystem::path get_user_documents_path();

// Json tools

/*! \brief Recursive get_or_default function for json*/
template <typename T>
T json_get_or_default(const json& data, T default_value, const char* key)
{
    return data.value(key, default_value);
}

template <typename T>
T json_get_or_default(const json& data, T default_value, const char* key, const char* keys...)
{
    try
    {
        return json_get_or_default<T>(data.at(key), default_value, keys);
    }
    catch (const json::exception&)
    {
        LOG_DEBUG("Missing key in json: {}", key);
        return default_value;
    }
}
} // namespace holovibes
