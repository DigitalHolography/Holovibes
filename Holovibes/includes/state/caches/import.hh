/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{

//! \brief First frame read
using StartFrame = UIntParameter<0, "start_frame">;
//! \brief Last frame read
using EndFrame = UIntParameter<0, "end_frame">;

using ImportCache = MicroCache<StartFrame, EndFrame>;

} // namespace holovibes
