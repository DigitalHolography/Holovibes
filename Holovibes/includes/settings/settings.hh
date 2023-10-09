#pragma once

/**
 * @file settings.hh
 * This file provides a class for every settings of holovibes.
 */

#include <string>

#define DECLARE_SETTING(name, type)                                                                                    \
    struct name                                                                                                        \
    {                                                                                                                  \
        type value;                                                                                                    \
    };                                                                                                                 \
                                                                                                                       \
    inline bool operator==(const name& lhs, const name& rhs) { return lhs.value == rhs.value; }

namespace holovibes::settings
{
    /// @brief The number of frames per seconds to load from a file
    /// or a camera to the GPU input buffer.
    DECLARE_SETTING(InputFPS, size_t)

    /// @brief The path of the input file.
    DECLARE_SETTING(InputFilePath, std::string)
}