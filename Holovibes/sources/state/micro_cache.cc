#include "micro_cache.hh"

namespace holovibes
{
MicroCache::MicroCache() { micro_caches_.insert(this); }

MicroCache::~MicroCache() { micro_caches_.erase(this); }

void MicroCache::synchronize()
{
    for (const auto& pair : to_update)
    {
        void* src = pair.second;
        void* dst = elem_to_ptr_[pair.first];
        size_t size = elem_to_size_[pair.first];
        memcpy(dst, src, size);
    }

    to_update.clear();
}
} // namespace holovibes
