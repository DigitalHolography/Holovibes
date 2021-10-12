#include "global_state_holder.hh"

namespace holovibes
{

GSH& GSH::instance()
{
    // Major issues can spawn here
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_ == nullptr)
        instance_ = new GSH();
    return *instance_;
}

} // namespace holovibes
