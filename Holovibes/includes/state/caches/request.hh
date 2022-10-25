/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{

using RequestClearImgAccu = TriggerParameter<"request_clear_img_accu">;
using RequestTimeTransformationCuts = BoolParameter<false, "request_time_transformation_cuts">;

using RequestCache = MicroCache<RequestClearImgAccu, RequestTimeTransformationCuts>;

} // namespace holovibes
