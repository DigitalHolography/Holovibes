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

#include "types.hh"

#include <filesystem>
#include <fstream>

#include "rect.hh"
#include "hardware_limits.hh"
#include "frame_desc.hh"
#include "cufft.h"

#include "logger.hh"

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
T read_file(const std::filesystem::path& path)
{
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (file.fail())
    {
        throw std::runtime_error("Could not read file " + path.string());
    }

    std::streampos end = file.tellg();
    file.seekg(0, std::ios::beg);
    std::streampos begin = file.tellg();

    T result;
    result.resize(static_cast<size_t>(end - begin));

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(result.data()), end - begin);
    file.close();

    return result;
}

namespace holovibes
{
/*! \brief return width and height with the same ratio and the max of the two being window_size */
void get_good_size(ushort& width, ushort& height, ushort window_size);
/*! \brief Returns the directory of the currently used executable file */
std::string get_exe_dir();
/*! \brief Return the first not used filename available from the parameter filename as a base */
std::string get_record_filename(std::string filename);
/*! \brief Returns the absolute path from a relative path (prepend by the execution directory) */
std::string create_absolute_path(const std::string& relative_path);
/*! \brief Returns the absolute path to the user Documents folder */
std::filesystem::path get_user_documents_path();
} // namespace holovibes
// Json tools
namespace holovibes
{

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
    catch (const json::exception& e)
    {
        LOG_DEBUG("Missing key in json: {}", key);
        return default_value;
    }
}
} // namespace holovibes

namespace image
{
template <typename T>
void grey_to_rgb_size(T& buffer_size)
{
    buffer_size *= 3;
}
} // namespace image
