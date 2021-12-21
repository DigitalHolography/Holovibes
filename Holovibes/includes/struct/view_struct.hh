#pragma once

#include <atomic>
// #include <boost/pfr/core.hpp>

#include "logger.hh"
#include "all_struct.hh"

typedef unsigned int uint;

namespace holovibes
{
// clang-format off
struct View_Window // : public json_struct
{
    // FIXME: remove slice in attr name
    bool log_scale_slice_enabled = false;
    bool contrast_enabled = false;
    bool contrast_auto_refresh = true;
    bool contrast_invert = false;
    float contrast_min = 1.f;
    float contrast_max = 65535.f;

    operator json() const
    {
        return json{
            {"log enabled", log_scale_slice_enabled},
            {"contrast", {
                    {"enabled", contrast_enabled},
                    {"auto refresh", contrast_auto_refresh},
                    {"invert", contrast_invert},
                    {"min", contrast_min},
                    {"max", contrast_max}
                }
            }
        };
    }

    View_Window() = default;

    explicit View_Window(const json& data)
        : log_scale_slice_enabled(data["log enabled"])
        , contrast_enabled(data["contrast"]["enabled"])
        , contrast_auto_refresh(data["contrast"]["auto refresh"])
        , contrast_invert(data["contrast"]["invert"])
        , contrast_min(data["contrast"]["min"])
        , contrast_max(data["contrast"]["max"])
    {
    }
};

struct View_XYZ : public View_Window
{
    bool flip_enabled = false;
    float rot = 0;

    uint img_accu_level = 1;

    operator json() const
    {
        json j = static_cast<View_Window>(*this);
        j["flip"] = flip_enabled;
        j["rot"] = rot;
        j["img accu level"] = img_accu_level;

        return j;
    }

    View_XYZ() = default;

    explicit View_XYZ(const json& data)
        : View_Window(data)
        , flip_enabled(data["flip"])
        , rot(data["rot"])
        , img_accu_level(data["img accu level"])
    {
    }
};

struct View_Accu // : public json_struct
{
    int accu_level = 0;

    operator json() const { return json{{"accu level", accu_level}}; }

    View_Accu() = default;

    explicit View_Accu(const json& data)
        : accu_level(data["accu level"])
    {
    }
};

struct View_PQ : public View_Accu
{
    uint index = 0;

    operator json() const
    {
        json j = static_cast<View_Accu>(*this);
        j["index"] = index;
        return j;
    }

    View_PQ() = default;

    explicit View_PQ(const json& data)
        : View_Accu(data)
        , index(data["index"])
    {
    }
};

struct View_XY : public View_Accu
{
    uint cuts = 0;

    operator json() const
    {
        json j = static_cast<View_Accu>(*this);
        j["cuts"] = cuts;
        return j;
    }

    View_XY() = default;

    explicit View_XY(const json& data)
        : View_Accu(data)
        , cuts(data["cuts"])
    {
    }
};

// clang-format on

inline std::ostream& operator<<(std::ostream& os, View_Window obj) { return os << json{obj}; }

inline std::ostream& operator<<(std::ostream& os, View_XYZ obj) { return os << json{obj}; }

inline std::ostream& operator<<(std::ostream& os, View_Accu obj) { return os << json{obj}; }

inline std::ostream& operator<<(std::ostream& os, View_XY obj) { return os << json{obj}; }

inline std::ostream& operator<<(std::ostream& os, View_PQ obj) { return os << json{obj}; }

} // namespace holovibes
