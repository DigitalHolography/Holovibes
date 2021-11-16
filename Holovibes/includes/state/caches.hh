#pragma once

#include "micro_cache.hh"

namespace holovibes
{
NEW_MICRO_CACHE(BatchCache, (uint, batch_size));

NEW_MICRO_CACHE(ComputeCache, (uint, batch_size), (uint, time_transformation_stride), (uint, time_transformation_size));

NEW_MICRO_CACHE(
    Filter2DCache, (int, filter2d_n1), (int, filter2d_n2), (bool, filter2d_enabled), (bool, filter2d_view_enabled));
} // namespace holovibes
