#include "global_state_holder.hh"

namespace holovibes
{

GSH& GSH::instance()
{
    static GSH instance_;
    return instance_;
}

GSH::synchronize()
{
    std::lock_guard<std::mutex> lock(mutex_);

    for (void* x : to_update)
    {
        for (void* y : cache_map[x].first)
            memccpy(y, x, 0, type_to_byte[cache_map[x].second]);
    }
}

} // namespace holovibes
