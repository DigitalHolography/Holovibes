#pragma once

#include <atomic>

typedef unsigned int uint;

struct Composite_P
{
    std::atomic<int> p_min{0};
    std::atomic<int> p_max{0};
};

struct Composite_RGB : public Composite_P
{
    std::atomic<float> weight_r{1};
    std::atomic<float> weight_g{1};
    std::atomic<float> weight_b{1};
};

struct Composite_hsv : public Composite_P
{
    std::atomic<float> slider_threshold_min{0.01f};
    std::atomic<float> slider_threshold_max{1.0f};
    std::atomic<float> low_threshold{0.2f};
    std::atomic<float> high_threshold{99.8f};
};

struct Composite_H : public Composite_hsv
{
    std::atomic<bool> blur_enabled{false};
    std::atomic<uint> blur_kernel_size{1};
};

struct Composite_SV : public Composite_hsv
{
    std::atomic<bool> p_activated{false};
};

struct Composite_HSV
{
    Composite_H h{};
    Composite_SV s{};
    Composite_SV v{};
};