#pragma once

#include <atomic>

#include "all_struct.hh"

typedef unsigned int uint;

namespace holovibes
{
struct Composite_P : public json_struct
{
    std::atomic<int> p_min{0};
    std::atomic<int> p_max{0};

    json to_json() const override { return json{"p", {{"min", p_min.load()}, {"max", p_max.load()}}}; }
};

struct Composite_RGB : public Composite_P
{
    std::atomic<float> weight_r{1};
    std::atomic<float> weight_g{1};
    std::atomic<float> weight_b{1};

    json to_json() const override
    {
        return json{
            Composite_P::to_json(),
            {"weight", {{"r", weight_r.load()}, {"g", weight_g.load()}, {"b", weight_b.load()}}},
        };
    }
};

struct Composite_hsv : public Composite_P
{
    std::atomic<float> slider_threshold_min{0.01f};
    std::atomic<float> slider_threshold_max{1.0f};
    std::atomic<float> low_threshold{0.2f};
    std::atomic<float> high_threshold{99.8f};

    json to_json() const override
    {
        return json{Composite_P::to_json(),
                    {"slider threshold", {{"min", slider_threshold_min.load()}, {"max", slider_threshold_max.load()}}},
                    {"threshold", {{"low", low_threshold.load()}, {"high", high_threshold.load()}}}};
    }
};

struct Composite_H : public Composite_hsv
{
    std::atomic<bool> blur_enabled{false};
    std::atomic<uint> blur_kernel_size{1};

    json to_json() const override
    {
        auto hsv = Composite_hsv::to_json();
        hsv["blur"] = json{{"enabled", blur_enabled.load()}, {"kernel size", blur_kernel_size.load()}};
        return hsv;
    }
};

struct Composite_SV : public Composite_hsv
{
    std::atomic<bool> p_activated{false};

    json to_json() const override
    {
        auto hsv = Composite_hsv::to_json();
        hsv["p"]["activated"] = p_activated.load();
        return hsv;
    }
};

struct Composite_HSV : public json_struct
{
    Composite_H h{};
    Composite_SV s{};
    Composite_SV v{};

    json to_json() const override { return json{{"h", h.to_json()}, {"s", s.to_json()}, {"v", v.to_json()}}; }
};
} // namespace holovibes