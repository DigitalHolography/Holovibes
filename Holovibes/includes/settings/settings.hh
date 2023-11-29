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
#include "enum/enum_space_transformation.hh"
#include "enum/enum_time_transformation.hh"
#include "enum/enum_computation.hh"
#include "rect.hh"

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
 * @brief The setting that specifies if we loop at the end of the
 * input_file once it has been read entirely.
 */
DECLARE_SETTING(LoopOnInputFile, bool)

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

//ex filter2d_cache
DECLARE_SETTING(Filter2dN1, int)
DECLARE_SETTING(Filter2dN2, int)
DECLARE_SETTING(Filter2dSmoothLow, int)
DECLARE_SETTING(Filter2dSmoothHigh, int)

//ex FileReadCache
/**
 * @brief The size of the buffer in CPU memory used to read a file
 * when `LoadFileInGPU` is set to false.
 */
DECLARE_SETTING(FileBufferSize, size_t)

//ex ImportCache
/*! \brief 
 * \param start_frame First frame read
 * \param end_frame Last frame read
 */

/**
 *@brief Index of the first frame to read from the input file (excluded).
 */
DECLARE_SETTING(InputFileStartIndex, size_t)

/**
 *@brief Index of the last frame to read from the input file (included).
 */
DECLARE_SETTING(InputFileEndIndex, size_t)


//ex ExportCache

/*! \brief Construct a new new micro cache object
 * \param frame_record_enabled Is holovibes currently recording
 * \param chart_record_enabled Enables the signal and noise chart record
 */
DECLARE_SETTING(FrameRecordEnabled, bool)
DECLARE_SETTING(ChartRecordEnabled, bool)

// ex Advanced Cache
DECLARE_SETTING(DisplayRate, float)
DECLARE_SETTING(InputBufferSize, size_t)
DECLARE_SETTING(RecordBufferSize, size_t)
DECLARE_SETTING(ContrastLowerThreshold, float)
DECLARE_SETTING(RawBitshift, size_t)
DECLARE_SETTING(ContrastUpperThreshold, float)
DECLARE_SETTING(RenormConstant, unsigned)
DECLARE_SETTING(CutsContrastPOffset, size_t)

// ex ComputeCache
DECLARE_SETTING(BatchSize, uint)
DECLARE_SETTING(TimeStride, uint)
DECLARE_SETTING(TimeTransformationSize, uint)
DECLARE_SETTING(SpaceTransformation, holovibes::SpaceTransformation)
DECLARE_SETTING(TimeTransformation, holovibes::TimeTransformation)
DECLARE_SETTING(Lambda, float)
DECLARE_SETTING(ZDistance, float)
DECLARE_SETTING(ConvolutionEnabled, bool)
DECLARE_SETTING(ConvolutionMatrix, std::vector<float>)
DECLARE_SETTING(DivideConvolutionEnabled, bool)
DECLARE_SETTING(ComputeMode, holovibes::Computation)
DECLARE_SETTING(PixelSize, float)
DECLARE_SETTING(UnwrapHistorySize, uint)
DECLARE_SETTING(IsComputationStopped, bool)
DECLARE_SETTING(TimeTransformationCutsOutputBufferSize, uint)
DECLARE_SETTING(FilterEnabled, bool)
DECLARE_SETTING(InputFilter, std::vector<float>)

/*! \brief ex ZoneCache
 *
 * \param signal_zone The zone for the nsignal chart
 * \param noise_zone The zone for the noise chart
 * \param composite_zone The area on which we'll normalize the colors
 * \param zoomed_zone The area used to limit the stft computations
 * \param reitcle_zone The zone of the reticle area
 */
DECLARE_SETTING(SignalZone, units::RectFd)
DECLARE_SETTING(NoiseZone, units::RectFd)
DECLARE_SETTING(CompositeZone, units::RectFd)
DECLARE_SETTING(ZoomedZone, units::RectFd)
DECLARE_SETTING(ReticleZone, units::RectFd)

} // namespace holovibes::settings