#pragma once

typedef unsigned int uint;

struct Composite_P
{
    int p_min = 0;
    int p_max = 0;
};

struct Composite_RGB : public Composite_P
{
    float weight_r = 1;
    float weight_g = 1;
    float weight_b = 1;
};

struct Composite_hsv : public Composite_P
{
    float slider_threshold_min = 0.01f;
    float slider_threshold_max = 1.0f;
    float low_threshold = 0.2f;
    float high_threshold = 99.8f;
};

struct Composite_H : public Composite_hsv
{
    bool blur_enabled = false;
    uint blur_kernel_size = 1;
};

struct Composite_SV : public Composite_hsv
{
    bool p_activated = false;
};

struct Composite_HSV
{
    Composite_H h{};
    Composite_SV s{};
    Composite_SV v{};
};

inline std::ostream& operator<<(std::ostream& os, Composite_P obj)
{
    return os << "obj.pmin : " << obj.p_min << " - obj.pmax : " << obj.p_max;
}

inline std::ostream& operator<<(std::ostream& os, Composite_RGB obj)
{
    return os << "obj.weight_r : " << obj.weight_g << " - obj.weight_g" << obj.weight_b << " - obj.weight_b"
              << static_cast<Composite_P>(obj);
}

inline std::ostream& operator<<(std::ostream& os, Composite_hsv obj) { return os; }

inline std::ostream& operator<<(std::ostream& os, Composite_H obj) { return os; }

inline std::ostream& operator<<(std::ostream& os, Composite_SV obj) { return os; }

inline std::ostream& operator<<(std::ostream& os, Composite_HSV obj) { return os; }