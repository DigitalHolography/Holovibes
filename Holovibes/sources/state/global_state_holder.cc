#include "global_state_holder.hh"

namespace holovibes
{

GSH& GSH::instance()
{
    static GSH instance_;
    return instance_;
}

} // namespace holovibes
