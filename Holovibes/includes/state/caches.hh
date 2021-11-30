#pragma once

#include "micro_cache.hh"
#include "enum_space_transformation.hh"
#include "enum_window_kind.hh"
#include "enum_img_type.hh"
#include "enum_computation.hh"
#include "enum_composite_kind.hh"
#include "view_struct.hh"

namespace holovibes
{
/*! \brief Construct a new new micro cache object
 * \param batch_size Size of BatchInputQueue's batches
 * \param time_transformation_stride Number of pipe iterations between two time transformations (STFT/PCA)
 * \param time_transformation_size Number of images used by the time transformation
 * \param space_transformation Space transformation algorithm to apply in hologram mode
 * \param time_transformation Time transformation to apply in hologram mode
 * \param lambda Wave length of the laser
 * \param z_distance z value used by fresnel transform
 * \param convolution_enabled Is convolution enabled
 * \param divide_convolution_enabled
 * \param input_fps The input FPS FIXME: move to ImportCache
 * \param compute_mode Mode of computation of the image
 */
NEW_INITIALIZED_MICRO_CACHE(ComputeCache,
                            (uint, batch_size, 1),
                            (uint, time_transformation_stride, 1),
                            (uint, time_transformation_size, 1),
                            (SpaceTransformation, space_transformation, SpaceTransformation::NONE),
                            (TimeTransformation, time_transformation, TimeTransformation::STFT),
                            (float, lambda, 852e-9f),
                            (float, z_distance, 1.50f),
                            (bool, convolution_enabled, false),
                            (bool, divide_convolution_enabled, false),
                            (uint, input_fps, 60),
                            (Computation, compute_mode, Computation::Raw));

/*! \brief Construct a new new micro cache object
 * \param composite_kind
 */
NEW_INITIALIZED_MICRO_CACHE(CompositeCache,
                            (CompositeKind, composite_kind, CompositeKind::RGB),
                            (bool, composite_auto_weights, false));

/*! \brief Construct a new new micro cache object
 * \param frame_record_enabled Is holovibes currently recording
 * \param chart_record_enabled Enables the signal and noise chart record
 */
NEW_INITIALIZED_MICRO_CACHE(ExportCache, (bool, frame_record_enabled, false), (bool, chart_record_enabled, false));

/*! \brief Construct a new new micro cache object
 * \param start_frame First frame read
 * \param start_frame Last frame read
 */
NEW_INITIALIZED_MICRO_CACHE(ImportCache, (uint, start_frame, 0), (uint, end_frame, 0));

/*! \brief Construct a new new micro cache object
 * \param img_type Type of the image displayed
 * \param x
 * \param y
 * \param p
 * \param q
 * \param xy
 * \param xz
 * \param yz
 * \param filter2d
 * \param current_window
 * \param lens_view_enabled
 * \param chart_display_enabled Enables the signal and noise chart display
 * \param filter2d_enabled Enables filter 2D
 * \param filter2d_view_enabled Enables filter 2D View
 * \param fft_shift_enabled Is shift fft enabled (switching representation diagram)
 * \param raw_view_enabled Display the raw interferogram when we are in hologram mode.
 * \param cuts_view_enabled Are slices YZ and XZ enabled
 */

NEW_INITIALIZED_MICRO_CACHE(ViewCache,
                            (ImgType, img_type, ImgType::Modulus),
                            (View_XY, x, View_XY{}),
                            (View_XY, y, View_XY{}),
                            (View_PQ, p, View_PQ{}),
                            (View_PQ, q, View_PQ{}),
                            (View_XYZ, xy, View_XYZ{}),
                            (View_XYZ, xz, View_XYZ{}),
                            (View_XYZ, yz, View_XYZ{}),
                            (View_Window, filter2d, View_Window{}),
                            (WindowKind, current_window, WindowKind::XYview),
                            (bool, lens_view_enabled, false),
                            (bool, chart_display_enabled, false),
                            (bool, filter2d_enabled, false),
                            (bool, filter2d_view_enabled, false),
                            (bool, fft_shift_enabled, false),
                            (bool, raw_view_enabled, false),
                            (bool, cuts_view_enabled, false));

/*! \brief Construct a new new micro cache object
 * \param filter2d_n1 Filter2D low radius
 * \param filter2d_n2 Filter2D high radius
 * \param filter2d_enabled Enables filter 2D
 * \param filter2d_view_enabled Enables filter 2D View
 */
NEW_INITIALIZED_MICRO_CACHE(Filter2DCache,
                            (int, filter2d_n1, 0),
                            (int, filter2d_n2, 1)); /*(bool, filter2d_enabled,), (bool, filter2d_view_enabled));*/

/*! \brief Construct a new new micro cache object
 *
 * \param display_rate of frame per seconds displayed
 * \param input_buffer_size Max size of input queue in number of images.
 */
NEW_INITIALIZED_MICRO_CACHE(AdvancedCache, (float, display_rate, 30), (uint, input_buffer_size, 256));

/*! \brief Construct a new new micro cache object
 *
 * \param file_buffer_size Max file buffer size
 */
NEW_INITIALIZED_MICRO_CACHE(FileReadCache, (uint, file_buffer_size, 32));
} // namespace holovibes
