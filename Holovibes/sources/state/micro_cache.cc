#include "micro_cache.hh"

#include <cstring>
#include "logger.hh"

namespace holovibes
{
MicroCache::MicroCache() { micro_caches_.insert(this); }

MicroCache::~MicroCache() { micro_caches_.erase(this); }
} // namespace holovibes
