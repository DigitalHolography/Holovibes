#pragma once

#include "pipe.hh"
#include "logger.hh"
#include "micro_cache.hh"

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
        LOG_DEBUG(compute_worker, "UPDATE RequestClearImgAccu");

        image_accumulation_->clear();
    }

    template <>
    void operator()<RequestTimeTransformationCuts>(bool new_value, bool old_value, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE RequestTimeTransformationCuts");

        if (new_value == false)
            dispose_cuts();
        else
            init_cuts();
    }
};
} // namespace holovibes
