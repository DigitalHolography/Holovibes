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
    
    bool exec_auto_contrast_ = false;
    bool get_exec_auto_contrast() { return exec_auto_contrast_; }
    void request_exec_auto_contrast() { exec_auto_contrast_ = true; }
    void reset_exec_auto_contrast() { exec_auto_contrast_ = false; }

    SERIALIZE_JSON_STRUCT(ViewWindow, log_enabled, contrast)
};

/*! \class ViewXYZ
 *
 * \brief Class that represents ViewXYZ
 */
struct ViewXYZ : public ViewWindow
{
    bool log_enabled = false;
    bool flip_enabled = false;
    float rotation = 0;
    unsigned img_accu_level = 1;

    bool operator!=(const ViewXYZ& rhs)
    {
        return View_Window::operator!=(rhs) || flip_enabled != rhs.flip_enabled || rotation != rhs.rotation ||
               img_accu_level != rhs.img_accu_level;
    }

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
struct View_PQ : public ViewAccu
{
    unsigned index = 0;

    bool operator!=(const View_PQ& rhs) { return ViewAccu::operator!=(rhs) || index != rhs.index; }

    SERIALIZE_JSON_STRUCT(View_PQ, accu_level, index)
};

/*! \class ViewXY
 *
 * \brief Class that represents ViewXY
 */
struct View_XY : public ViewAccu
{
    unsigned cuts = 0;

    bool operator!=(const View_XY& rhs) { return ViewAccu::operator!=(rhs) || cuts != rhs.cuts; }

    SERIALIZE_JSON_STRUCT(View_XY, accu_level, cuts)
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
