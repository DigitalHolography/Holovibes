#include "parameters_handler.hh"
#include "logger.hh"

namespace holovibes
{
void ParametersHandlerCache::synchronize()
{
    LOG_DEBUG(main, "++++++++++++++ TRY TO SYNC");
    for (auto change : change_pool)
    {
        LOG_DEBUG(main, "++++++++++++++ SYNC {} WITH {}", change.param_to_change->get_key(), change.ref->get_key());
        change.param_to_change->sync_with(change.ref);
    }
}
} // namespace holovibes
