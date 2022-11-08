/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "parameter.hh"
#include "micro_cache.hh"

#include "rect.hh"

namespace holovibes
{
template <StringLiteral Key>
using RectFdParameter = Parameter<units::RectFd, DefaultLiteral<units::RectFd>{}, Key>;

// clang-format off

//! \brief The zone for the nsignal chart
class SignalZone : public RectFdParameter<"signal_zone">{};
//! \brief The zone for the noise chart
class NoiseZone : public RectFdParameter<"noise_zone">{};
//! \brief The area on which we'll normalize the colors
class CompositeZone : public RectFdParameter<"composite_zone">{};
//! \brief The area used to limit the stft computations
class ZoomedZone : public RectFdParameter<"zoomed_zone">{};
//! \brief The zone of the reticle area
class ReticleZone : public RectFdParameter<"reticle_zone">{};

// clang-format on

using ZoneCache = MicroCache<SignalZone, NoiseZone, CompositeZone, ZoomedZone, ReticleZone>;

} // namespace holovibes
