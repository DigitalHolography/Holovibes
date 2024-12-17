/*!
 * \file settings.hh
 * \brief This file contains the definition of all settings of Holovibes. Each settings is defined by a struct holding a
 * single variable `value`.
 */
#pragma once

#include <string>
#include <optional>
#include "enum/enum_record_mode.hh"
#include "enum/enum_import_type.hh"
#include "struct/view_struct.hh"
#include "struct/composite_struct.hh"
#include "enum/enum_window_kind.hh"
#include "enum/enum_camera_kind.hh"
#include "enum/enum_space_transformation.hh"
#include "enum/enum_time_transformation.hh"
#include "enum/enum_computation.hh"
#include "enum/enum_device.hh"
#include "enum/enum_recorded_data_type.hh"
#include "rect.hh"
#include "frame_desc.hh"

#define DECLARE_SETTING(name, type)                                                                                    \
    struct name                                                                                                        \
    {                                                                                                                  \
        type value;                                                                                                    \
    };                                                                                                                 \
                                                                                                                       \
    inline bool operator==(const name& lhs, const name& rhs) { return lhs.value == rhs.value; }

namespace holovibes::settings
{

#pragma region Input

DECLARE_SETTING(ImportType, holovibes::ImportType);

/*!
 * \brief The number of frames per seconds to load from a file
 * or a camera to the GPU input buffer.
 */
DECLARE_SETTING(InputFPS, size_t);

DECLARE_SETTING(InputBufferSize, size_t);

DECLARE_SETTING(CameraFps, uint);
DECLARE_SETTING(PixelSize, float);
DECLARE_SETTING(CameraKind, holovibes::CameraKind);

/*!
 * \brief The size of the buffer in CPU memory used to read a file
 * when `LoadFileInGPU` is set to false.
 */
DECLARE_SETTING(FileBufferSize, size_t);

/*! \brief The path of the input file. */
DECLARE_SETTING(InputFilePath, std::string);
DECLARE_SETTING(ImportedFileFd, camera::FrameDescriptor);

/*! \brief Index of the first frame to read from the input file (excluded). */
DECLARE_SETTING(InputFileStartIndex, size_t);

/*! \brief Index of the last frame to read from the input file (included). */
DECLARE_SETTING(InputFileEndIndex, size_t);

/*!
 * \brief The setting that specifies if we load input file entirely in GPU
 * before sending it to the compute pipeline input queue.
 */
DECLARE_SETTING(LoadFileInGPU, bool);

#pragma endregion

#pragma region Record
/*!
 * \brief The setting that specifies the path of the file where to record
 * the frames.
 */
DECLARE_SETTING(RecordFilePath, std::string);

/*! \brief The setting that specifies the number of frames to record. */
DECLARE_SETTING(RecordFrameCount, std::optional<size_t>);

/*! \brief Specifies the number of frames to skip before recording. */
DECLARE_SETTING(RecordFrameOffset, size_t);
DECLARE_SETTING(FrameSkip, uint);

DECLARE_SETTING(Mp4Fps, uint);

/*! \brief The setting that specifies the mode of the record. */
DECLARE_SETTING(RecordMode, holovibes::RecordMode);
DECLARE_SETTING(DataType, holovibes::RecordedDataType);

DECLARE_SETTING(RecordQueueLocation, holovibes::Device);
DECLARE_SETTING(RecordBufferSize, size_t);

/*! \brief Is holovibes currently recording */
DECLARE_SETTING(FrameRecordEnabled, bool);

/*! \brief Enables the signal and noise chart record */
DECLARE_SETTING(ChartRecordEnabled, bool);

#pragma endregion

#pragma region Chart Record

/*! \brief The zone for the nsignal chart */
DECLARE_SETTING(SignalZone, units::RectFd);

/*! \brief The zone for the noise chart */
DECLARE_SETTING(NoiseZone, units::RectFd);

DECLARE_SETTING(ChartDisplayEnabled, bool);

#pragma endregion

#pragma region Filter2D

DECLARE_SETTING(Filter2dEnabled, bool);
DECLARE_SETTING(Filter2dN1, int);
DECLARE_SETTING(Filter2dN2, int);
DECLARE_SETTING(Filter2dSmoothLow, int);
DECLARE_SETTING(Filter2dSmoothHigh, int);

DECLARE_SETTING(FilterFileName, std::string);
DECLARE_SETTING(FilterEnabled, bool);
DECLARE_SETTING(InputFilter, std::vector<float>);

#pragma endregion

#pragma region Contrast

DECLARE_SETTING(ContrastLowerThreshold, float);
DECLARE_SETTING(ContrastUpperThreshold, float);
DECLARE_SETTING(CutsContrastPOffset, size_t);

/*! \brief The zone of the reticle area */
DECLARE_SETTING(ReticleZone, units::RectFd);
DECLARE_SETTING(ReticleScale, float);
DECLARE_SETTING(ReticleDisplayEnabled, bool);

#pragma endregion

#pragma region Convolution

DECLARE_SETTING(ConvolutionEnabled, bool);
DECLARE_SETTING(ConvolutionMatrix, std::vector<float>);
DECLARE_SETTING(DivideConvolutionEnabled, bool);
DECLARE_SETTING(ConvolutionFileName, std::string);

#pragma endregion

#pragma region Window

DECLARE_SETTING(DisplayRate, float);

DECLARE_SETTING(XY, ViewXYZ);
DECLARE_SETTING(XZ, ViewXYZ);
DECLARE_SETTING(YZ, ViewXYZ);
DECLARE_SETTING(Filter2d, ViewWindow);

DECLARE_SETTING(CurrentWindow, holovibes::WindowKind);

DECLARE_SETTING(LensViewEnabled, bool);
DECLARE_SETTING(Filter2dViewEnabled, bool);
DECLARE_SETTING(RawViewEnabled, bool);
DECLARE_SETTING(CutsViewEnabled, bool);

#pragma endregion

#pragma region Space Tansform

DECLARE_SETTING(SpaceTransformation, holovibes::SpaceTransformation);

DECLARE_SETTING(BatchSize, uint);

DECLARE_SETTING(Lambda, float);
DECLARE_SETTING(ZDistance, float);

#pragma endregion

#pragma region Time Transformation

DECLARE_SETTING(TimeTransformation, holovibes::TimeTransformation);
DECLARE_SETTING(TimeTransformationSize, uint);

DECLARE_SETTING(X, ViewXY);
DECLARE_SETTING(Y, ViewXY);
DECLARE_SETTING(P, ViewPQ);
DECLARE_SETTING(Q, ViewPQ);

DECLARE_SETTING(TimeTransformationCutsOutputBufferSize, uint);

#pragma endregion

#pragma region Post Process

DECLARE_SETTING(FftShiftEnabled, bool);

DECLARE_SETTING(RenormEnabled, bool);
DECLARE_SETTING(RenormConstant, unsigned);

DECLARE_SETTING(RegistrationEnabled, bool);
DECLARE_SETTING(RegistrationZone, float);

DECLARE_SETTING(RawBitshift, size_t);

#pragma endregion

#pragma region Compute

/*! \brief The setting that specifies the size of the output buffer. */
DECLARE_SETTING(OutputBufferSize, size_t);

/*! \name View Cache */
/*! \brief The setting that specifies the type of the image displayed. */
DECLARE_SETTING(ImageType, ImgType);

/*! \name ComputeCache */
DECLARE_SETTING(TimeStride, uint);

DECLARE_SETTING(ComputeMode, holovibes::Computation);
DECLARE_SETTING(IsComputationStopped, bool);

#pragma endregion

#pragma region Composite

/*! \brief The area on which we'll normalize the colors */
DECLARE_SETTING(CompositeZone, units::RectFd);

/*! \name CompositeCache */
DECLARE_SETTING(CompositeKind, holovibes::CompositeKind);
DECLARE_SETTING(CompositeAutoWeights, bool);
DECLARE_SETTING(RGB, holovibes::CompositeRGB);
DECLARE_SETTING(HSV, holovibes::CompositeHSV);
DECLARE_SETTING(ZFFTShift, bool);

#pragma endregion

DECLARE_SETTING(BenchmarkMode, bool);

} // namespace holovibes::settings
