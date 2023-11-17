#pragma once

/**
 * @file settings.hh
 * This file provides a class for every settings of holovibes.
 */

#include <string>
#include <optional>
#include "enum/enum_record_mode.hh"
#include "struct/view_struct.hh"
#include "enum/enum_window_kind.hh"

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
 *@brief Index of the first frame to read from the input file (excluded).
 */
DECLARE_SETTING(InputFileStartIndex, size_t)

/**
 *@brief Index of the last frame to read from the input file (included).
 */
DECLARE_SETTING(InputFileEndIndex, size_t)

/**
 * @brief The setting that specifies if we load input file entirely in GPU
 * before sending it to the compute pipeline input queue.
 */
DECLARE_SETTING(LoadFileInGPU, bool)

/**
 * @brief The setting that specifies the path of the file where to record
 * the frames.
 */
DECLARE_SETTING(RecordFilePath, std::string)

/**
 * @brief The setting that specifies the number of frames to record.
 */
DECLARE_SETTING(RecordFrameCount, std::optional<size_t>)

/**
 * @brief The setting that specifies the mode of the record.
 */
DECLARE_SETTING(RecordMode, holovibes::RecordMode)

/**
 * @brief The setting that specifies the number of frames to skip before
 * starting the record.
 */
DECLARE_SETTING(RecordFrameSkip, size_t)

/**
 * @brief The setting that specifies the size of the output buffer.
 */
DECLARE_SETTING(OutputBufferSize, size_t)

/**
 * @brief The setting that specifies whether the batch mode is enabled or not.
 * If it is enabled, a batch script is read and executed.
 */
DECLARE_SETTING(BatchEnabled, bool)

/**
 * @brief The setting that specifies the path of the batch script to execute.
 */
DECLARE_SETTING(BatchFilePath, std::string)

// ex view_cache
/**
 * @brief The setting that specifies the type of the image displayed.
 */
DECLARE_SETTING(ImageType, ImgType)
DECLARE_SETTING(X, ViewXY)
DECLARE_SETTING(Y, ViewXY)
DECLARE_SETTING(P, ViewPQ)
DECLARE_SETTING(Q, ViewPQ)
DECLARE_SETTING(XY, ViewXYZ)
DECLARE_SETTING(XZ, ViewXYZ)
DECLARE_SETTING(YZ, ViewXYZ)
DECLARE_SETTING(Filter2d, ViewWindow)
DECLARE_SETTING(CurrentWindow, holovibes::WindowKind)
DECLARE_SETTING(LensViewEnabled, bool)
DECLARE_SETTING(ChartDisplayEnabled, bool)
DECLARE_SETTING(Filter2dEnabled, bool)
DECLARE_SETTING(Filter2dViewEnabled, bool)
DECLARE_SETTING(FftShiftEnabled, bool)
DECLARE_SETTING(RawViewEnabled, bool)
DECLARE_SETTING(CutsViewEnabled, bool)
DECLARE_SETTING(RenormEnabled, bool)
DECLARE_SETTING(ReticleScale, float)
DECLARE_SETTING(ReticleDisplayEnabled, bool)

} // namespace holovibes::settings