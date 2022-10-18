/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

#include "rect.hh"

namespace holovibes
{

// For non constexpr type, we need to do this in order to get a default value
struct RectFdLiteral
{
    operator units::RectFd() const { return units::RectFd{}; }
    static constexpr RectFdLiteral instance() { return RectFdLiteral(); }
};

template <StringLiteral Key>
using RectFdParameter = CustomParameter<units::RectFd, RectFdLiteral::instance(), Key>;

//! \brief The zone for the nsignal chart
using SignalZone = RectFdParameter<"signal_zone">;
//! \brief The zone for the noise chart
using NoiseZone = RectFdParameter<"noise_zone">;
//! \brief The area on which we'll normalize the colors
using CompositeZone = RectFdParameter<"composite_zone">;
//! \brief The area used to limit the stft computations
using ZoomedZone = RectFdParameter<"zoomed_zone">;
//! \brief The zone of the reticle area
using ReticleZone = RectFdParameter<"reticle_zone">;

using ZoneCache = MicroCache<SignalZone, NoiseZone, CompositeZone, ZoomedZone, ReticleZone>;

} // namespace holovibes
