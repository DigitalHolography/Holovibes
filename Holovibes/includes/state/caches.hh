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


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// TODO: IN API set_batch_size remove gsh call
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
