#pragma once

#include "micro_cache.hh"
#include "enum_space_transformation.hh"
#include "enum_img_type.hh"

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
 */
NEW_INITIALIZED_MICRO_CACHE(ComputeCache,
                            (uint, batch_size, 1),
                            (uint, time_transformation_stride, 1),
                            (uint, time_transformation_size, 1),
                            (SpaceTransformation, space_transformation, SpaceTransformation::NONE),
                            (TimeTransformation, time_transformation, TimeTransformation::STFT),
                            (float, lambda, 852e-9f),
                            (float, z_distance, 1.50f));

/*! \brief Construct a new new micro cache object
 * \param img_type Type of the image displayed
 */
NEW_INITIALIZED_MICRO_CACHE(ViewCache, (ImgType, img_type, ImgType::Modulus));

/*! \brief Construct a new new micro cache object
 * \param filter2d_n1 Filter2D low radius
 * \param filter2d_n2 Filter2D high radius
 * \param filter2d_enabled Enables filter 2D
 * \param filter2d_view_enabled Enables filter 2D View
 */
NEW_INITIALIZED_MICRO_CACHE(Filter2DCache,
                            (int, filter2d_n1, 0),
                            (int, filter2d_n2, 1)); /*(bool, filter2d_enabled,), (bool, filter2d_view_enabled));*/
} // namespace holovibes
