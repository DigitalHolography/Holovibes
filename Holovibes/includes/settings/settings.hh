/*!
 * \file settings.hh
 * \brief This file contains the definition of all settings of Holovibes. Each settings is defined by a struct holding a
 * single variable `value`.
 */

#pragma once

#include <string>
#include <optional>
#include "enum_file_load_kind.hh"
#include "enum/enum_record_mode.hh"
#include "enum/enum_import_type.hh"
#include "struct/view_struct.hh"
#include "struct/composite_struct.hh"
#include "enum/enum_window_kind.hh"
#include "enum/enum_camera_kind.hh"
#include "enum_recorded_eye_type.hh"
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
/*!
 * \brief The number of frames per seconds to load from a file
 * or a camera to the GPU input buffer.
 */
DECLARE_SETTING(InputFPS, size_t);

/*!
 * \brief The path of the input file.
 */
DECLARE_SETTING(InputFilePath, std::string);
DECLARE_SETTING(InputFd, camera::FrameDescriptor);

DECLARE_SETTING(ImportType, holovibes::ImportType);
DECLARE_SETTING(CameraKind, holovibes::CameraKind);

/*!
 * \brief The setting that specifies how to read frames from a file.
 */
DECLARE_SETTING(FileLoadKind, holovibes::FileLoadKind);

/*!
 * \brief The setting that specifies the path of the file where to record
 * the frames.
 */
DECLARE_SETTING(RecordFilePath, std::string);

/*!
 * \brief The setting that specifies the number of frames to record.
 */
DECLARE_SETTING(RecordFrameCount, std::optional<size_t>);

/*!
 * \brief The setting that specifies the mode of the record.
 */
DECLARE_SETTING(RecordMode, holovibes::RecordMode);

/*!
 * \brief Get the eye that is recorded by the program
 * Can be LEFT, RIGHT or NONE if no eye is selected
 * It only influences the name of the output files and is saved across .holo files
 */
DECLARE_SETTING(RecordedEye, RecordedEyeType);

/*!
 * \brief The setting that specifies the number of frames to skip before
 * starting the record.
 */
DECLARE_SETTING(RecordFrameOffset, size_t);

/*! \brief The setting that specifies the size of the output buffer. */
DECLARE_SETTING(OutputBufferSize, size_t);

/*! \name View Cache */
/*! \brief The setting that specifies the type of the image displayed. */
DECLARE_SETTING(ImageType, ImgType);
DECLARE_SETTING(Unwrap2d, bool);

DECLARE_SETTING(X, ViewXY);
DECLARE_SETTING(Y, ViewXY);
DECLARE_SETTING(P, ViewPQ);
DECLARE_SETTING(Q, ViewPQ);

DECLARE_SETTING(XY, ViewXYZ);
DECLARE_SETTING(XZ, ViewXYZ);
DECLARE_SETTING(YZ, ViewXYZ);
DECLARE_SETTING(Filter2d, ViewWindow);

DECLARE_SETTING(XYContrastRange, ContrastRange);
DECLARE_SETTING(XZContrastRange, ContrastRange);
DECLARE_SETTING(YZContrastRange, ContrastRange);
DECLARE_SETTING(Filter2dContrastRange, ContrastRange);

DECLARE_SETTING(CurrentWindow, holovibes::WindowKind);
DECLARE_SETTING(LensViewEnabled, bool);
DECLARE_SETTING(ChartDisplayEnabled, bool);
DECLARE_SETTING(Filter2dEnabled, bool);
DECLARE_SETTING(Filter2dViewEnabled, bool);
DECLARE_SETTING(FftShiftEnabled, bool);
DECLARE_SETTING(RegistrationEnabled, bool);
DECLARE_SETTING(RawViewEnabled, bool);
DECLARE_SETTING(CutsViewEnabled, bool);
DECLARE_SETTING(RenormEnabled, bool);
DECLARE_SETTING(ReticleScale, float);
DECLARE_SETTING(RegistrationZone, float);
DECLARE_SETTING(ReticleDisplayEnabled, bool);

/*! \name Filter2D Cache */
DECLARE_SETTING(Filter2dN1, int);
DECLARE_SETTING(Filter2dN2, int);
DECLARE_SETTING(Filter2dSmoothLow, int);
DECLARE_SETTING(Filter2dSmoothHigh, int);
DECLARE_SETTING(FilterFileName, std::string);

/*! \name FileReadCache */
/*!
 * \brief The size of the buffer in CPU memory used to read a file
 * when `FileLoadKind` is not set to GPU.
 */
DECLARE_SETTING(FileBufferSize, size_t);

/*! \name Import Cache */
/*! \brief Index of the first frame to read from the input file (excluded). */
DECLARE_SETTING(InputFileStartIndex, size_t);

/*! \brief Index of the last frame to read from the input file (included). */
DECLARE_SETTING(InputFileEndIndex, size_t);

/*! \name Export Cache */
/*! \brief Is acquisition (writing to the record_queue) of frames activated. */
DECLARE_SETTING(FrameAcquisitionEnabled, bool);

/*! \brief Enables the signal and noise chart record */
DECLARE_SETTING(ChartRecordEnabled, bool);

/*! \name Advanced Cache */
DECLARE_SETTING(DisplayRate, float);
DECLARE_SETTING(InputBufferSize, size_t);
DECLARE_SETTING(RecordBufferSize, size_t);
DECLARE_SETTING(ContrastLowerThreshold, float);
DECLARE_SETTING(RawBitshift, size_t);
DECLARE_SETTING(ContrastUpperThreshold, float);
DECLARE_SETTING(RenormConstant, unsigned);
DECLARE_SETTING(CutsContrastPOffset, size_t);
DECLARE_SETTING(BenchmarkMode, bool);

/*! \name ComputeCache */
DECLARE_SETTING(BatchSize, uint);
DECLARE_SETTING(TimeStride, uint);
DECLARE_SETTING(TimeTransformationSize, uint);
DECLARE_SETTING(SpaceTransformation, holovibes::SpaceTransformation);
DECLARE_SETTING(TimeTransformation, holovibes::TimeTransformation);
DECLARE_SETTING(Lambda, float);
DECLARE_SETTING(ZDistance, float);

DECLARE_SETTING(ConvolutionMatrix, std::vector<float>);
DECLARE_SETTING(DivideConvolutionEnabled, bool);
DECLARE_SETTING(ConvolutionFileName, std::string);

DECLARE_SETTING(ComputeMode, holovibes::Computation);
DECLARE_SETTING(PixelSize, float);
DECLARE_SETTING(IsComputationStopped, bool);
DECLARE_SETTING(TimeTransformationCutsOutputBufferSize, uint);
DECLARE_SETTING(InputFilter, std::vector<float>);

/*! \name ZoneCache */
/*! \brief The zone for the nsignal chart */
DECLARE_SETTING(SignalZone, units::RectFd);

/*! \brief The zone for the noise chart */
DECLARE_SETTING(NoiseZone, units::RectFd);

/*! \brief The area on which we'll normalize the colors */
DECLARE_SETTING(CompositeZone, units::RectFd);

/*! \brief The zone of the reticle area */
DECLARE_SETTING(ReticleZone, units::RectFd);

/*! \name CompositeCache */
DECLARE_SETTING(CompositeKind, holovibes::CompositeKind);
DECLARE_SETTING(CompositeAutoWeights, bool);
DECLARE_SETTING(RGB, holovibes::CompositeRGB);
DECLARE_SETTING(HSV, holovibes::CompositeHSV);
DECLARE_SETTING(ZFFTShift, bool);

DECLARE_SETTING(RecordQueueLocation, holovibes::Device);

DECLARE_SETTING(FrameSkip, uint);
DECLARE_SETTING(Mp4Fps, uint);
DECLARE_SETTING(CameraFps, uint);

DECLARE_SETTING(DataType, holovibes::RecordedDataType);
} // namespace holovibes::settings
