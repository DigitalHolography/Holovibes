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
/**
 * @brief The number of frames per seconds to load from a file
 * or a camera to the GPU input buffer.
 */
DECLARE_SETTING(InputFPS, size_t)

/**
 * @brief The path of the input file.
 */
DECLARE_SETTING(InputFilePath, std::string)

/**
 * @brief The size of the buffer in CPU memory used to read a file
 * when `LoadFileInGPU` is set to false.
 */
DECLARE_SETTING(FileBufferSize, size_t)

/**
 * @brief The setting that specifies if we loop at the end of the
 * input_file once it has been read entirely.
 */
DECLARE_SETTING(LoopOnInputFile, bool)

/**
 *@brief Index of the last frame to read from the input file (excluded).
 */
DECLARE_SETTING(EndIndex, size_t)

/**
 *@brief Index of the first frame to read from the input file (excluded).
 */
DECLARE_SETTING(StartIndex, size_t)

/**
 * @brief The setting that specifies if we load input file entirely in GPU
 * before sending it to the compute pipeline input queue.
 */
DECLARE_SETTING(LoadFileInGPU, bool)
} // namespace holovibes::settings