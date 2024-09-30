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

/*! \class Internals
 *
 * \brief Class that represents the internals cache
 */
struct Internals
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

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(Zones, signal_zone, noise_zone, composite_zone, zoomed_zone, reticle_zone);
    };

    /*! \class Record
     *
     * \brief Class that represents Record
     */
    struct Record
    {
        float input_fps = 10000.0f;
        unsigned record_start_frame = 0;
        unsigned record_end_frame = 0;
        bool frame_record_enabled = false;
        bool chart_record_enabled = false;

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(
            Record, input_fps, record_start_frame, record_end_frame, frame_record_enabled, chart_record_enabled);
    };

    /*! \class Enabled
     *
     * \brief Class that represents Enabled
     */
    struct Enabled
    {
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

            /*! \brief Will be expanded into `to_json` and `from_json` functions. */
            SERIALIZE_JSON_STRUCT(ViewEnabled, lens, filter2d, raw, cuts);
        };

        bool filter2d = false;
        bool chart = false;
        bool fft_shift = false;
        ViewEnabled views;

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(Enabled, filter2d, chart, fft_shift, views);
    };

    /*! \class Misc
     *
     * \brief Class that represents Misc
     */
    struct Misc
    {
        float pixel_size = 12.0f;
        unsigned unwrap_history_size = 1;
        bool is_computation_stopped = true;
        int raw_bitshift = 0;

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(Misc, pixel_size, unwrap_history_size, is_computation_stopped);
    };

    Zones zones;
    Record record;
    Enabled enabled;
    Misc misc;

    std::vector<float> convo_matrix;
    WindowKind current_window = WindowKind::XYview;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(Internals, zones, record, enabled, misc, convo_matrix, current_window);
};

} // namespace holovibes