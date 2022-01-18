#pragma once

#include "logger.hh"
#include "all_struct.hh"
#include "enum_img_type.hh"

typedef unsigned int uint;

namespace holovibes
{
struct ViewContrast
{
    bool enabled = false;
    bool auto_refresh = true;
    bool invert = false;
    float min = 1.f;
    float max = 65535.f;

    SERIALIZE_JSON_STRUCT(ViewContrast, enabled, auto_refresh, invert, min, max)
};

struct ViewWindow
{
    bool log_enabled = false;
    ViewContrast contrast;

    SERIALIZE_JSON_STRUCT(ViewWindow, log_enabled, contrast)
};

struct ViewXYZ : public ViewWindow
{
    bool flip_enabled = false;
    float rot = 0;
    unsigned img_accu_level = 1;

    SERIALIZE_JSON_STRUCT(ViewXYZ, log_enabled, contrast, flip_enabled, rot, img_accu_level)
};

struct ViewAccu
{
    int accu_level = 0;

    SERIALIZE_JSON_STRUCT(ViewAccu, accu_level)
};

struct ViewPQ : public ViewAccu
{
    unsigned index = 0;

    SERIALIZE_JSON_STRUCT(ViewPQ, accu_level, index)
};

struct ViewXY : public ViewAccu
{
    unsigned cuts = 0;

    SERIALIZE_JSON_STRUCT(ViewXY, accu_level, cuts)
};

struct Windows
{
    ViewXYZ xy;
    ViewXYZ yz;
    ViewXYZ xz;
    ViewWindow filter2d;

    SERIALIZE_JSON_STRUCT(Windows, xy, yz, xz, filter2d);
};

struct Reticle
{
    bool display_enabled = false;
    float reticle_scale = 0.5f;

    SERIALIZE_JSON_STRUCT(Reticle, display_enabled, reticle_scale);
};

struct Views
{
    ImgType img_type = ImgType::Modulus;
    bool fft_shift = false;
    ViewXY x;
    ViewXY y;
    ViewPQ p;
    ViewPQ q;
    Windows window;
    bool renorm = false;
    Reticle reticle;

    SERIALIZE_JSON_STRUCT(Views, img_type, fft_shift, x, y, p, q, window, renorm, reticle);
};

} // namespace holovibes
