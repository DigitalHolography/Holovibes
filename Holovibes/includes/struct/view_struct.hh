/*! \file
 *
 * \brief View structure
 *
 */

#pragma once

#include "logger.hh"
#include "all_struct.hh"
#include "enum_img_type.hh"

#define CONSTRUCTOR(name, arg_name)

typedef unsigned int uint;

namespace holovibes
{
/*! \class ViewContrast
 *
 * \brief Class that represents ViewContrast
 */
struct ViewContrast
{
    bool enabled = false;
    bool auto_refresh = true;
    bool invert = false;
    float min = 1.f;
    float max = 65535.f;

    SERIALIZE_JSON_STRUCT(ViewContrast, enabled, auto_refresh, invert, min, max)
};

/*! \class ViewWindow
 *
 * \brief Class that represents ViewWindow
 */
struct ViewWindow
{
    bool log_enabled = false;

    ViewContrast contrast;

    SERIALIZE_JSON_STRUCT(ViewWindow, log_enabled, contrast)
};

/*! \class ViewXYZ
 *
 * \brief Class that represents ViewXYZ
 */
struct ViewXYZ : public ViewWindow
{
    bool flip_enabled = false;
    float rot = 0;
    unsigned img_accu_level = 1;

    SERIALIZE_JSON_STRUCT(ViewXYZ, log_enabled, contrast, flip_enabled, rot, img_accu_level)
};

/*! \class ViewAccu
 *
 * \brief Class that represents ViewAccu
 */
struct ViewAccu
{
    int accu_level = 0;

    SERIALIZE_JSON_STRUCT(ViewAccu, accu_level)
};

/*! \class ViewPQ
 *
 * \brief Class that represents ViewPQ
 */
struct View_Accu_PQ : public ViewAccu
{
    unsigned index = 0;

    SERIALIZE_JSON_STRUCT(View_Accu_PQ, accu_level, index)
};

/*! \class ViewXY
 *
 * \brief Class that represents ViewXY
 */
struct View_Accu_XY : public ViewAccu
{
    unsigned cuts = 0;

    SERIALIZE_JSON_STRUCT(View_Accu_XY, accu_level, cuts)
};

/*! \class Windows
 *
 * \brief Class that represents the Windows
 */
struct Windows
{
    ViewXYZ xy;
    ViewXYZ yz;
    ViewXYZ xz;
    ViewWindow filter2d;

    SERIALIZE_JSON_STRUCT(Windows, xy, yz, xz, filter2d);

    void Update();
    void Load();
};

/*! \class Reticle
 *
 * \brief Class that represents the Reticle
 */
struct Reticle
{
    bool display_enabled = false;
    float reticle_scale = 0.5f;

    SERIALIZE_JSON_STRUCT(Reticle, display_enabled, reticle_scale);

    void Update();
    void Load();
};

/*! \class View
 *
 * \brief Class that represents the view cache
 */
struct Views
{
    ImgType img_type = ImgType::Modulus;
    bool fft_shift = false;
    View_Accu_XY x;
    View_Accu_XY y;
    View_Accu_PQ p;
    View_Accu_PQ q;
    Windows window;
    bool renorm = false;
    Reticle reticle;

    SERIALIZE_JSON_STRUCT(Views, img_type, fft_shift, x, y, p, q, window, renorm, reticle);

    void Update();
    void Load();
};

} // namespace holovibes
