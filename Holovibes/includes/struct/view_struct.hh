#pragma once

#include <atomic>

typedef unsigned int uint;

struct WindowView
{
    std::atomic<bool> flip_enabled{false};
    std::atomic<float> rot{0};
    std::atomic<bool> log_scale_slice_enabled{false};
    std::atomic<bool> img_acc_slice_enabled{false};
    std::atomic<uint> img_acc_slice_level{1};
    std::atomic<bool> contrast_enabled{false};
    std::atomic<bool> contrast_auto_refresh{true};
    std::atomic<bool> contrast_invert{false};
    std::atomic<float> contrast_min_slice{1.f};
    std::atomic<float> contrast_max_slice{65535.f};
};
