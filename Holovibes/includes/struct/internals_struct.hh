#pragma once

#include "all_struct.hh"
#include "enum_window_kind.hh"
#include "rect.hh"

namespace holovibes
{

struct Zones
{
    RectFd signal_zone;
    RectFd noise_zone;
    RectFd composite_zone;
    RectFd zoomed_zone;
    RectFd reticle_zone;
};

struct Record
{
    float input_fps = 60f;
    unsigned record_start_frame = 0;
    unsigned record_end_frame = 0;
    bool frame_record_enabled = false;
    bool chart_record_enabled = false;
};

struct ViewEnabled
{
    bool lens = false;
    bool filter2d = false;
    bool raw = false;
    bool cuts = false;
};

struct Enabled
{
    bool filter2d = false;
    bool chart = false;
    bool fft_shift = false;
    ViewEnabled views;
};

struct Misc
{
    float pixel_size = 12.0f;
    unsigned unwrap_history_size = 1;
    bool is_computation_stopped = true;
};

struct Internals
{
    Zones zones;
    Record record;
    Enabled enabled;
    Misc misc;

    std::vector<float> convo_matrix;
    WindowKind current_window = WindowKind::XYview;
};

SERIALIZE_JSON_FWD(Zones)
SERIALIZE_JSON_FWD(Record)
SERIALIZE_JSON_FWD(ViewEnabled)
SERIALIZE_JSON_FWD(Enabled)
SERIALIZE_JSON_FWD(Misc)
SERIALIZE_JSON_FWD(Internals)

} // namespace holovibes