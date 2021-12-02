#pragma once

#include <atomic>

#include "logger.hh"
#include "all_struct.hh"

typedef unsigned int uint;

namespace holovibes
{
struct View_Window // : public json_struct
{
    // FIXME: remove slice in attr name
    std::atomic<bool> log_scale_slice_enabled{false};

    std::atomic<bool> contrast_enabled{false};
    std::atomic<bool> contrast_auto_refresh{true};
    std::atomic<bool> contrast_invert{false};
    std::atomic<float> contrast_min{1.f};
    std::atomic<float> contrast_max{65535.f};

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
    std::atomic<bool> flip_enabled{false};
    std::atomic<float> rot{0};

    std::atomic<uint> img_accu_level{1};

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
    std::atomic<int> accu_level{1};

    json to_json() const { return json{"accu level", accu_level.load()}; }

    void from_json(const json& data) { accu_level = data["accu level"]; }
};

struct View_PQ : public View_Accu
{
    std::atomic<uint> index{0};

    json to_json() const { return json{View_Accu::to_json(), {"index", index.load()}}; }

    void from_json(const json& data)
    {
        View_Accu::from_json(data);
        index = data["index"];
    }
};

struct View_XY : public View_Accu
{
    std::atomic<uint> cuts{0};

    json to_json() const { return json{View_Accu::to_json(), {"cuts", cuts.load()}}; }

    void from_json(const json& data)
    {
        View_Accu::from_json(data);
        cuts = data["cuts"];
    }
};
} // namespace holovibes