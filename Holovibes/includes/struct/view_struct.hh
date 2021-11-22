#pragma once

#include <atomic>
// #include <boost/pfr/core.hpp>

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

    std::atomic<uint> img_accu_level{1};
};

struct View_Accu
{
    int accu_level = 1;
    inline View_Accu& operator=(const View_Accu& accu)
    {
        accu_level = accu.accu_level;
        return *this;
    };
};

struct View_PQ : public View_Accu
{
    std::atomic<uint> index{0};
};

struct View_XY : public View_Accu
{
    uint cuts = 0;
    inline View_XY& operator=(const View_XY& x)
    {
        cuts = x.cuts;
        return *this;
    };
};

inline std::ostream& operator<<(std::ostream& os, View_Accu obj) { return os << obj.accu_level; }

inline std::ostream& operator<<(std::ostream& os, View_XY obj) { return os << obj.cuts; }

// inline std::ostream& operator<<(std::ostream& os, View_Accu obj)
// {
//     boost::pfr::for_each_field(obj, [](auto& field) {
//         os << field;
//     }
//     return os;
// }
