#pragma once

#include <atomic>

#include "all_struct.hh"

typedef unsigned int uint;

namespace holovibes
{
// clang-format off
struct Composite_P // : public json_struct
{
    int p_min = 0;
    int p_max = 0;

    operator json() const { return json{{"min", p_min}, {"max", p_max}}; }

    Composite_P() = default;

    explicit Composite_P(const json& data)
    {
        const json& p_data = data["p"];
        p_min = p_data["min"];
        p_max = p_data["max"];
    }
};

struct Composite_RGB : public Composite_P
{
    float weight_r = 1;
    float weight_g = 1;
    float weight_b = 1;

    operator json() const
    {
        return json{
            {"p", static_cast<Composite_P>(*this)},
            {"weight", {
                    {"r", weight_r},
                    {"g", weight_g},
                    {"b", weight_b}
                }
            }
        };
    }

    Composite_RGB() = default;

    explicit Composite_RGB(const json& data)
        : Composite_P(data)
    {
        auto weight_data = data["weight"];
        weight_r = weight_data["r"];
        weight_g = weight_data["g"];
        weight_b = weight_data["b"];
    }
};

struct Composite_hsv : public Composite_P
{
    float slider_threshold_min{0.01f};
    float slider_threshold_max{1.0f};
    float low_threshold{0.2f};
    float high_threshold{99.8f};

    operator json() const
    {
        return json{
            {"p", static_cast<Composite_P>(*this)},
            {"slider threshold", {
                    {"min", slider_threshold_min},
                    {"max", slider_threshold_max}
                }
            },
            {"threshold", {
                    {"low", low_threshold},
                    {"high", high_threshold}
                }
            }
        };
    }

    Composite_hsv() = default;

    explicit Composite_hsv(const json& data)
        : Composite_P(data)
    {
        const json& slider_data = data["slider threshold"];
        slider_threshold_min = slider_data["min"];
        slider_threshold_max = slider_data["max"];

        const json& threshold_data = data["threshold"];
        low_threshold = threshold_data["low"];
        high_threshold = threshold_data["high"];
    }
};

struct Composite_H : public Composite_hsv
{
    bool blur_enabled = false;
    uint blur_kernel_size = 1;

    operator json() const
    {
        json j = static_cast<Composite_hsv>(*this);
        j["blur"] = json{
            {"enabled", blur_enabled},
            {"kernel size", blur_kernel_size}
        };
        return j;
    }

    Composite_H() = default;

    explicit Composite_H(const json& data)
        : Composite_hsv(data)
    {
        const json& blur_data = data["blur"];
        blur_enabled = blur_data["enabled"];
        blur_kernel_size = blur_data["kernel size"];
    }
};

struct Composite_SV : public Composite_hsv
{
    bool p_activated = false;

    operator json() const
    {
        json j = static_cast<Composite_hsv>(*this);
        j["p"]["activated"] = p_activated;
        return j;
    }

    Composite_SV() = default;

    explicit Composite_SV(const json& data)
        : Composite_hsv(data)
        , p_activated(data["p"]["activated"])
    {
    }
};

struct Composite_HSV // : public json_struct
{
    Composite_H h{};
    Composite_SV s{};
    Composite_SV v{};

    operator json() const { return json{{"h", h}, {"s", s}, {"v", v}}; }

    Composite_HSV() = default;

    explicit Composite_HSV(const json& data)
        : h(data["h"])
        , s(data["s"])
        , v(data["v"])
    {
    }
};
// clang-format on
} // namespace holovibes
