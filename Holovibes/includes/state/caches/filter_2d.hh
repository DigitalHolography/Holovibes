/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{

//! \brief Low radius
using Filter2DN1 = IntParameter<0, "filter2d_n1">;
//! \brief High radius
using Filter2DN2 = IntParameter<1, "filter2d_n2">;
//! \brief Low smoothing // May be moved in filter2d Struct
using Filter2DSmoothLow = IntParameter<0, "filter2d_smooth_low">;
//! \brief High smoothing
using Filter2DSmoothHigh = IntParameter<0, "filter2d_smooth_high">;

using Filter2DCache = MicroCache<Filter2DN1, Filter2DN2, Filter2DSmoothLow, Filter2DSmoothHigh>;

} // namespace holovibes
