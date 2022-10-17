/*! \file
 *
 * \brief #TODO Add a description for this file
 */

/*! \brief Construct a new micro cache object
 *
 * \param signal_zone The zone for the nsignal chart
 * \param noise_zone The zone for the noise chart
 * \param composite_zone The area on which we'll normalize the colors
 * \param zoomed_zone The area used to limit the stft computations
 * \param reitcle_zone The zone of the reticle area
 */
NEW_INITIALIZED_MICRO_CACHE(ZoneCache,
                            (units::RectFd, signal_zone, units::RectFd{}),
                            (units::RectFd, noise_zone, units::RectFd{}),
                            (units::RectFd, composite_zone, units::RectFd{}),
                            (units::RectFd, zoomed_zone, units::RectFd{}),
                            (units::RectFd, reticle_zone, units::RectFd{}))

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{

//! \brief The zone for the nsignal chart
using SignalZone = CustomParameter<units::RectFd, units::RectFd{}, "signal_zone">;
//! \brief The zone for the noise chart
using NoiseZone = CustomParameter<units::RectFd, units::RectFd{}, "noise_zone">;
//! \brief The area on which we'll normalize the colors
using CompositeZone = CustomParameter<units::RectFd, units::RectFd{}, "composite_zone">;
//! \brief The area used to limit the stft computations
using ZoomedZone = CustomParameter<units::RectFd, units::RectFd{}, "zoomed_zone">;
//! \brief The zone of the reticle area
using ReticleZone = CustomParameter<units::RectFd, units::RectFd{}, "reticle_zone">;

using ZoneCache = MicroCache<SignalZone, NoiseZone, CompositeZone, ZoomedZone, ReticleZone>;

} // namespace holovibes
