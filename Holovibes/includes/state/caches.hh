/*! \file
 *
 * \brief declaration of all micro caches
 * 
 */

#pragma once

#include "micro_cache.hh"
#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_window_kind.hh"
#include "enum_img_type.hh"
#include "enum_computation.hh"
#include "enum_composite_kind.hh"
#include "view_struct.hh"
#include "composite_struct.hh"
#include "rect.hh"

namespace holovibes
{
/*! \brief Construct a new new micro cache object
 *
 * \param display_rate of frame per seconds displayed
 * \param input_buffer_size Max size of input queue in number of images.
 * \param record_buffer_size Max size of frame record queue in number of images.
 * \param output_buffer_size Max size of output queue in number of images.
 * \param contrast_lower_threshold
 * \param contrast_upper_threshold
 */
NEW_INITIALIZED_MICRO_CACHE(AdvancedCache,
                            (float, display_rate, 30),
                            (uint, input_buffer_size, 512),
                            (uint, output_buffer_size, 256),
                            (uint, record_buffer_size, 1024),
                            (float, contrast_lower_threshold, 0.5f),
                            (unsigned int, raw_bitshift, 0),
                            (float, contrast_upper_threshold, 99.5f),
                            (unsigned, renorm_constant, 5),
                            (uint, cuts_contrast_p_offset, 2));

/*! \brief Construct a new new micro cache object
 * \param batch_size Size of BatchInputQueue's batches
 * \param time_stride Number of pipe iterations between two time transformations (STFT/PCA)
 * \param time_transformation_size Number of images used by the time transformation
 * \param space_transformation Space transformation algorithm to apply in hologram mode
 * \param time_transformation Time transformation to apply in hologram mode
 * \param lambda Wave length of the laser
 * \param z_distance z value used by fresnel transform
 * \param convolution_enabled Is convolution enabled
 * \param convo_matrix Input matrix used for convolution
 * \param divide_convolution_enabled
 * \param compute_mode Mode of computation of the image
 * \param pixel_size Size of a pixel in micron. Depends on camera or input file.
 * \param unwrap_history_size Max size of unwrapping corrections in number of images.
 * Determines how far, meaning how many iterations back, phase corrections
 * are taken in order to be applied to the current phase image.
 * \param is_computation_stopped Is the computation stopped
 * \param renorm_enabled Postprocessing renorm enabled
 * \param time_transformation_cuts_output_buffer_size Max size of time transformation cuts queue in number of images.
 * \param renorm_constant postprocessing remormalize multiplication constant
 */
NEW_INITIALIZED_MICRO_CACHE(ComputeCache,
                            (uint, batch_size, 1),
                            (uint, time_stride, 1),
                            (uint, time_transformation_size, 1),
                            (SpaceTransformation, space_transformation, SpaceTransformation::NONE),
                            (TimeTransformation, time_transformation, TimeTransformation::NONE),
                            (float, lambda, 852e-9f),
                            (float, z_distance, 1.50f),
                            (bool, convolution_enabled, false),
                            (std::vector<float>, convo_matrix, {}),
                            (bool, divide_convolution_enabled, false),
                            (Computation, compute_mode, Computation::Raw),
                            (float, pixel_size, 12.0f),
                            (uint, unwrap_history_size, 1),
                            (bool, is_computation_stopped, true),
                            (uint, time_transformation_cuts_output_buffer_size, 512));

/*! \brief Construct a new new micro cache object
 * \param composite_kind
 * \param composite_auto_weights
 * \param rgb
 * \param hsv
 */
NEW_INITIALIZED_MICRO_CACHE(CompositeCache,
                            (CompositeKind, composite_kind, CompositeKind::RGB),
                            (bool, composite_auto_weights, false),
                            (CompositeRGB, rgb, CompositeRGB{}),
                            (CompositeHSV, hsv, CompositeHSV{}));

/*! \brief Construct a new new micro cache object
 * \param frame_record_enabled Is holovibes currently recording
 * \param chart_record_enabled Enables the signal and noise chart record
 */
NEW_INITIALIZED_MICRO_CACHE(ExportCache, (bool, frame_record_enabled, false), (bool, chart_record_enabled, false));

/*! \brief Construct a new new micro cache object
 * \param start_frame First frame read
 * \param end_frame Last frame read
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
 * \param reticle_scale Reticle border scale
 * \param reticle_display_enabled Is the reticle overlay enabled
 */

NEW_INITIALIZED_MICRO_CACHE(ViewCache,
                            (ImgType, img_type, ImgType::Modulus),
                            (ViewXY, x, ViewXY{}),
                            (ViewXY, y, ViewXY{}),
                            (ViewPQ, p, ViewPQ{}),
                            (ViewPQ, q, ViewPQ{}),
                            (ViewXYZ, xy, ViewXYZ{}),
                            (ViewXYZ, xz, ViewXYZ{}),
                            (ViewXYZ, yz, ViewXYZ{}),
                            (ViewWindow, filter2d, ViewWindow{}),
                            (WindowKind, current_window, WindowKind::XYview),
                            (bool, lens_view_enabled, false),
                            (bool, chart_display_enabled, false),
                            (bool, filter2d_enabled, false),
                            (bool, filter2d_view_enabled, false),
                            (bool, fft_shift_enabled, false),
                            (bool, raw_view_enabled, false),
                            (bool, cuts_view_enabled, false),
                            (bool, renorm_enabled, true),
                            (float, reticle_scale, 0.5f),
                            (bool, reticle_display_enabled, false));

/*! \brief Construct a new new micro cache object
 * \param filter2d_n1 Filter2D low radius
 * \param filter2d_n2 Filter2D high radius
 * \param filter2d_enabled Enables filter 2D
 * \param filter2d_view_enabled Enables filter 2D View
 * \param filter2d_smooth_low Filter2D low smoothing // May be moved in filter2d Struct
 * \param filter2d_smooth_high Filter2D high smoothing
 */
NEW_INITIALIZED_MICRO_CACHE(Filter2DCache,
                            (int, filter2d_n1, 0),
                            (int, filter2d_n2, 1),
                            (int, filter2d_smooth_low, 0),
                            (int, filter2d_smooth_high, 0));
/*(bool, filter2d_enabled,), (bool, filter2d_view_enabled));*/

/*! \brief Construct a new new micro cache object
 *
 * \param file_buffer_size Max file buffer size
 */
NEW_INITIALIZED_MICRO_CACHE(FileReadCache, (uint, file_buffer_size, 512));

/*! \brief Construct a new micro cache object
 *
 * \param signal_zone The zone for the nsignal chart
 * \param noise_zone The zone for the noise chart
 * \param composite_zone The area on which we'll normalize the colors
 * \param zoomed_zone The area used to limit the stft computations
 * \param reitcle_zone The zone of the reticle area
 */
NEW_INITIALIZED_MICRO_CACHE(ZoneCache,
                            (units::RectFd, signal_zone, units::RectFd{}),
                            (units::RectFd, noise_zone, units::RectFd{}),
                            (units::RectFd, composite_zone, units::RectFd{}),
                            (units::RectFd, zoomed_zone, units::RectFd{}),
                            (units::RectFd, reticle_zone, units::RectFd{}))

} // namespace holovibes
