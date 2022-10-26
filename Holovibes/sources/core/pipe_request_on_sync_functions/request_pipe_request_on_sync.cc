#pragma once

#include "API.hh"

namespace holovibes
{
template <>
void RequestPipeRequestOnSync::operator()<RequestClearImgAccu>(TriggerRequest new_value,
                                                               TriggerRequest old_value,
                                                               Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE RequestClearImgAccu");

    image_accumulation_->clear();
}
} // namespace holovibes
