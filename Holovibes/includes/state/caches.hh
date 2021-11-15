#pragma once

#include "micro_cache.hh"

namespace holovibes::caches
{
NEW_MICRO_CACHE(BatchCache, (uint, batch_size));
} // namespace holovibes::caches
