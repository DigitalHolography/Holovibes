#pragma once

#include "all_struct.hh"
#include "enum_window_kind.hh"
#include "rect.hh"

namespace holovibes
{

struct Internals
{
    struct Zones
    {
        units::RectFd signal_zone;
        units::RectFd noise_zone;
        units::RectFd composite_zone;
        units::RectFd zoomed_zone;
        units::RectFd reticle_zone;

        SERIALIZE_JSON_STRUCT(Zones, signal_zone, noise_zone, composite_zone, zoomed_zone, reticle_zone)
    };

    struct Record
    {
        float input_fps = 60.0f;
        unsigned record_start_frame = 0;
        unsigned record_end_frame = 0;
        bool frame_record_enabled = false;
        bool chart_record_enabled = false;

        SERIALIZE_JSON_STRUCT(
            Record, input_fps, record_start_frame, record_end_frame, frame_record_enabled, chart_record_enabled)
    };

    struct Enabled
    {
        struct ViewEnabled
        {
            bool lens = false;
            bool filter2d = false;
            bool raw = false;
            bool cuts = false;

            SERIALIZE_JSON_STRUCT(ViewEnabled, lens, filter2d, raw, cuts)
        };

        bool filter2d = false;
        bool chart = false;
        bool fft_shift = false;
        ViewEnabled views;

        SERIALIZE_JSON_STRUCT(Enabled, filter2d, chart, fft_shift, views)
    };

    struct Misc
    {
        float pixel_size = 12.0f;
        unsigned unwrap_history_size = 1;
        bool is_computation_stopped = true;
        int raw_bitshift = 0;

        SERIALIZE_JSON_STRUCT(Misc, pixel_size, unwrap_history_size, is_computation_stopped)
    };

    Zones zones;
    Record record;
    Enabled enabled;
    Misc misc;

    std::vector<float> convo_matrix;
    WindowKind current_window = WindowKind::XYview;

    SERIALIZE_JSON_STRUCT(Internals, zones, record, enabled, misc, convo_matrix, current_window)
};

} // namespace holovibes