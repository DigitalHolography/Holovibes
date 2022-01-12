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
};

struct ViewWindow
{
    bool log_enabled = false;
    ViewContrast contrast;
};

struct ViewXYZ : public ViewWindow
{
    bool flip_enabled = false;
    float rot = 0;
    unsigned img_accu_level = 1;
};

struct ViewAccu
{
    int accu_level = 0;
};

struct ViewPQ : public ViewAccu
{
    unsigned index = 0;
};

struct ViewXY : public ViewAccu
{
    unsigned cuts = 0;
};

struct Windows
{
    ViewXYZ xy;
    ViewXYZ yz;
    ViewXYZ xz;
    ViewWindow filter2d;
};

struct Reticle
{
    bool display_enabled = false;
    float reticle_scale = 0.5f;
};

struct Views
{
    ImgType img_type = ImgType::MODULUS;
    bool fft_shift = false;
    ViewXY x;
    ViewXY y;
    ViewPQ p;
    ViewPQ q;
    Windows window;
    bool renorm = false;
    Reticle reticle;
};

// Forward declaration for to_json() and from_json()
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(ViewContrast)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(ViewWindow)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(ViewXYZ)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(ViewAccu)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(ViewPQ)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(ViewXY)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Windows)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Reticle)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Views)

} // namespace holovibes
