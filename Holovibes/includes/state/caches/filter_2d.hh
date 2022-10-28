/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{
// clang-format off

//! \brief Low radius
class Filter2DN1 : public IntParameter<0, "filter2d_n1">{};
//! \brief High radius
class Filter2DN2 : public IntParameter<1, "filter2d_n2">{};
//! \brief Low smoothing // May be moved in filter2d Struct
class Filter2DSmoothLow : public IntParameter<0, "filter2d_smooth_low">{};
//! \brief High smoothing
class Filter2DSmoothHigh : public IntParameter<0, "filter2d_smooth_high">{};

// clang-format on

class Filter2DCache : public MicroCache<Filter2DN1, Filter2DN2, Filter2DSmoothLow, Filter2DSmoothHigh>
{
};

} // namespace holovibes
