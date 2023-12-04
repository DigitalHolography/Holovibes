/*! \file
 *
 * \brief declaration of all micro caches
 * 
 */

#pragma once

#include "micro_cache.hh"
#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_window_kind.hh"
#include "enum_img_type.hh"
#include "enum_computation.hh"
#include "enum_composite_kind.hh"
#include "view_struct.hh"
#include "composite_struct.hh"
#include "rect.hh"

namespace holovibes
{
/*! \brief Construct a new new micro cache object
 * \param composite_kind
 * \param composite_auto_weights
 * \param rgb
 * \param hsv
 */

NEW_INITIALIZED_MICRO_CACHE(ExportCache, (bool, frame_record_enabled, false), (bool, chart_record_enabled, false), (std::optional<size_t>, nb_frame, std::nullopt));
} // namespace holovibes
