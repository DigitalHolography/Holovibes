#pragma once

#include <atomic>

typedef unsigned int uint;

struct Composite_RGB
{
    std::atomic<int> p_min{0};
    std::atomic<int> p_max{0};

    std::atomic<float> weight_r{1};
    std::atomic<float> weight_g{1};
    std::atomic<float> weight_b{1};
};

struct HSV_struct
{
    std::atomic<int> p_min{0};
    std::atomic<int> p_max{0};

    std::atomic<float> slider_threshold_min{0.01f};
    std::atomic<float> slider_threshold_max{1.0f};
    std::atomic<float> low_threshold{0.2f};
    std::atomic<float> high_threshold{99.8f};
};

struct H_struct : public HSV_struct
{
    std::atomic<bool> blur_enabled{false};
    std::atomic<uint> blur_kernel_size{1};
};

struct SV_struct : public HSV_struct
{
    std::atomic<bool> p_activated{false};
};

struct Composite_HSV
{
    H_struct h{};
    SV_struct s{};
    SV_struct v{};
};