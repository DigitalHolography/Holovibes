#pragma once

#include "micro_cache.hh"

namespace holovibes
{
NEW_MICRO_CACHE(BatchCache, (uint, batch_size));
NEW_MICRO_CACHE(BatchCache, (uint, batch_size), (uint, time_transformation_size), (uint, time_transformation_stride));
} // namespace holovibes
