#pragma once

#include <atomic>
// #include <boost/pfr/core.hpp>

#include "logger.hh"
#include "all_struct.hh"

typedef unsigned int uint;

namespace holovibes
{
struct View_Window // : public json_struct
{
    // FIXME: remove slice in attr name
    bool log_scale_slice_enabled = false;
    bool contrast_enabled = false;
    bool contrast_auto_refresh = true;
    bool contrast_invert = false;
    float contrast_min = 1.f;
    float contrast_max = 65535.f;

    json to_json() const
    {
        return json{{"log enabled", log_scale_slice_enabled.load()},
                    {"contrast",
                     {{"enabled", contrast_enabled.load()},
                      {"auto refresh", contrast_auto_refresh.load()},
                      {"invert", contrast_invert.load()},
                      {"min", contrast_min.load()},
                      {"max", contrast_max.load()}}}};
    }

    void from_json(const json& data)
    {
        log_scale_slice_enabled = data["log enabled"];

        const json& contrast_data = data["contrast"];
        contrast_enabled = contrast_data["enabled"];
        contrast_auto_refresh = contrast_data["auto refresh"];
        contrast_invert = contrast_data["invert"];
        contrast_min = contrast_data["min"];
        contrast_max = contrast_data["max"];
    }
};

struct View_XYZ : public View_Window
{
    bool flip_enabled = false;
    float rot = 0;

    uint img_accu_level = 1;

    json to_json() const
    {
        auto j = json{View_Window::to_json()};
        j["flip"] = flip_enabled.load();
        j["rot"] = rot.load();
        j["img accu level"] = img_accu_level.load();

        return j;
    }

    void from_json(const json& data)
    {
        View_Window::from_json(data);
        flip_enabled = data["flip"];
        rot = data["rot"];
        img_accu_level = data["img accu level"];
    }
};

struct View_Accu // : public json_struct
{
    int accu_level = 0;

    json to_json() const { return json{"accu level", accu_level.load()}; }

    void from_json(const json& data) { accu_level = data["accu level"]; }
};

struct View_PQ : public View_Accu
{
    uint index = 0;

    json to_json() const { return json{View_Accu::to_json(), {"index", index.load()}}; }

    void from_json(const json& data)
    {
        View_Accu::from_json(data);
        index = data["index"];
    }
};

struct View_XY : public View_Accu
{
    uint cuts = 0;

    json to_json() const { return json{View_Accu::to_json(), {"cuts", cuts.load()}}; }

    void from_json(const json& data)
    {
        View_Accu::from_json(data);
        cuts = data["cuts"];
    }
};
} // namespace holovibes

inline std::ostream& operator<<(std::ostream& os, View_Window obj)
{
    return os << obj.log_scale_slice_enabled << obj.contrast_enabled << obj.contrast_auto_refresh << obj.contrast_invert
              << obj.contrast_min << obj.contrast_max;
}

inline std::ostream& operator<<(std::ostream& os, View_XYZ obj)
{
    return os << '{' << obj.flip_enabled << ',' << obj.rot << ',' << obj.img_accu_level << ',' << std::boolalpha
              << obj.log_scale_slice_enabled << '}';
}

inline std::ostream& operator<<(std::ostream& os, View_Accu obj) { return os << obj.accu_level; }

inline std::ostream& operator<<(std::ostream& os, View_XY obj) { return os << obj.cuts; }

inline std::ostream& operator<<(std::ostream& os, View_PQ obj) { return os << obj.index; }
