#pragma once

#include "micro_cache.hh"
#include "enum_space_transformation.hh"

namespace holovibes
{
NEW_MICRO_CACHE(BatchCache, (uint, batch_size));

/*! \brief Construct a new new micro cache object
 * \param batch_size Size of BatchInputQueue's batches
 * \param time_transformation_stride Number of pipe iterations between two time transformations (STFT/PCA)
 * \param time_transformation_size Number of images used by the time transformation
 * \param space_transformation Space transformation algorithm to apply in hologram mode
 * \param time_transformation Time transformation to apply in hologram mode
 */
NEW_MICRO_CACHE(ComputeCache,
                (uint, batch_size),
                (uint, time_transformation_stride),
                (uint, time_transformation_size),
                (SpaceTransformation, space_transformation),
                (TimeTransformation, time_transformation));

/*! \brief Construct a new new micro cache object
 * \param filter2d_n1 Filter2D low radius
 * \param filter2d_n2 Filter2D high radius
 * \param filter2d_enabled Enables filter 2D
 * \param filter2d_view_enabled Enables filter 2D View
 */
NEW_MICRO_CACHE(
    Filter2DCache, (int, filter2d_n1), (int, filter2d_n2), (bool, filter2d_enabled), (bool, filter2d_view_enabled));
} // namespace holovibes
