#pragma once

#include "micro_cache.hh"
#include "pipe.hh"
#include "logger.hh"

namespace holovibes
{
class RequestPipeRequest
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

    template <>
    void operator()<RequestClearImgAccu>(TriggerParameter new_value, TriggerParameter old_value, Pipe& pipe)
    {
        LOG_TRACE(compute_worker, "UPDATE RequestClearImgAccu");

        image_accumulation_->clear();
    }
};
} // namespace holovibes
