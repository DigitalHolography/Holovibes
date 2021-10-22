#pragma once

#include <atomic>

typedef unsigned int uint;

struct View_Window
{
    // FIXME: remove slice in attr name
    std::atomic<bool> log_scale_slice_enabled{false};

    std::atomic<bool> contrast_enabled{false};
    std::atomic<bool> contrast_auto_refresh{true};
    std::atomic<bool> contrast_invert{false};
    std::atomic<float> contrast_min{1.f};
    std::atomic<float> contrast_max{65535.f};
};

struct View_XYZ : public View_Window
{
    std::atomic<bool> flip_enabled{false};
    std::atomic<float> rot{0};

    // FIXME: remove slice from name
    std::atomic<bool> img_accu_slice_enabled{false};
    std::atomic<uint> img_accu_slice_level{1};
};

struct View_Accu
{
    std::atomic<bool> accu_enabled{false};
    std::atomic<int> accu_level{1};
};

struct View_PQ : public View_Accu
{
    std::atomic<uint> index{0};
};

struct View_XY : public View_Accu
{
    std::atomic<uint> cuts{0};
};