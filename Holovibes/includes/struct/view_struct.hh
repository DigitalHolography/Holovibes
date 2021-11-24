#pragma once

#include <atomic>
// #include <boost/pfr/core.hpp>

typedef unsigned int uint;

struct View_Window
{
    // FIXME: remove slice in attr name
    bool log_scale_slice_enabled = false;

    bool contrast_enabled = false;
    bool contrast_auto_refresh = true;
    bool contrast_invert = false;
    float contrast_min = 1.f;
    float contrast_max = 65535.f;
};

struct View_XYZ : public View_Window
{
    bool flip_enabled = false;
    float rot = 0;

    uint img_accu_level = 1;
};

struct View_Accu
{
    int accu_level = 0;
};

struct View_PQ : public View_Accu
{
    uint index = 0;
};

struct View_XY : public View_Accu
{
    uint cuts = 0;
};

inline std::ostream& operator<<(std::ostream& os, View_Window obj)
{
    return os << obj.log_scale_slice_enabled << obj.contrast_enabled << obj.contrast_auto_refresh << obj.contrast_invert
              << obj.contrast_min << obj.contrast_max;
}

inline std::ostream& operator<<(std::ostream& os, View_XYZ obj)
{
    return os << obj.flip_enabled << obj.rot << obj.img_accu_level;
}

inline std::ostream& operator<<(std::ostream& os, View_Accu obj) { return os << obj.accu_level; }

inline std::ostream& operator<<(std::ostream& os, View_XY obj) { return os << obj.cuts; }

inline std::ostream& operator<<(std::ostream& os, View_PQ obj) { return os << obj.index; }

// inline std::ostream& operator<<(std::ostream& os, View_Accu obj)
// {
//     boost::pfr::for_each_field(obj, [](auto& field) {
//         os << field;
//     }
//     return os;
// }
