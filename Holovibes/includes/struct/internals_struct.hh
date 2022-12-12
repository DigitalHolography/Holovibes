/*! \file
 *
 * \brief Internals Struct
 *
 */

#pragma once

#include "all_struct.hh"
#include "enum_window_kind.hh"
#include "rect.hh"
namespace holovibes
{

/*! \class Zones
 *
 * \brief Class that represents Zones
 */
struct Zones
{
    units::RectFd signal_zone;
    units::RectFd noise_zone;
    units::RectFd composite_zone;
    units::RectFd zoomed_zone;
    units::RectFd reticle_zone;

    SERIALIZE_JSON_STRUCT(Zones, signal_zone, noise_zone, composite_zone, zoomed_zone, reticle_zone)
};

/*! \class RecordJsonStruct
 *
 * \brief Class that represents RecordJsonStruct
 */
struct RecordJsonStruct
{
    float input_fps = 60.0f;
    unsigned record_start_frame = 0;
    unsigned record_end_frame = 0;
    bool frame_record_enabled = false;
    bool chart_record_enabled = false;

    SERIALIZE_JSON_STRUCT(
        RecordJsonStruct, input_fps, record_start_frame, record_end_frame, frame_record_enabled, chart_record_enabled)
};

/*! \class ViewEnabled
 *
 * \brief Class that represents ViewEnabled
 */
struct ViewEnabled
{
    bool lens = false;
    bool filter2d = false;
    bool raw = false;
    bool cuts = false;

    SERIALIZE_JSON_STRUCT(ViewEnabled, lens, filter2d, raw, cuts)
};

/*! \class Enabled
 *
 * \brief Class that represents Enabled
 */
struct Enabled
{

    bool filter2d = false;
    bool chart = false;
    bool fft_shift = false;
    ViewEnabled views;

    SERIALIZE_JSON_STRUCT(Enabled, filter2d, chart, fft_shift, views)
};

/*! \class Misc
 *
 * \brief Class that represents Misc
 */
struct Misc
{
    float pixel_size = 12.0f;
    unsigned unwrap_history_size = 1;
    bool is_ComputeModeEnum_stopped = true;
    int raw_bitshift = 0;

    SERIALIZE_JSON_STRUCT(Misc, pixel_size, unwrap_history_size, is_ComputeModeEnum_stopped)
};

/*! \class Internals
 *
 * \brief Class that represents the internals cache
 */
struct Internals
{

    Zones zones;
    RecordJsonStruct record;
    Enabled enabled;
    Misc misc;

    std::vector<float> convo_matrix;
    WindowKind current_window = WindowKind::ViewXY;

    SERIALIZE_JSON_STRUCT(Internals, zones, record, enabled, misc, convo_matrix, current_window)
};

inline std::ostream& operator<<(std::ostream& os, const Internals& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const Zones& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const RecordJsonStruct& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const Enabled& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const Misc& value) { return os << json{value}; }
} // namespace holovibes