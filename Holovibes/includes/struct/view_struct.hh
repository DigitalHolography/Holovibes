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
    bool horizontal_flip = false;
    float rotation = 0;
    unsigned output_image_accumulation = 1;

    bool operator!=(const ViewXYZ& rhs)
    {
        return View_Window::operator!=(rhs) || horizontal_flip != rhs.horizontal_flip || rotation != rhs.rotation ||
               output_image_accumulation != rhs.output_image_accumulation;
    }

    SERIALIZE_JSON_STRUCT(ViewXYZ, log_enabled, contrast, horizontal_flip, rotation, output_image_accumulation)
};

/*! \class ViewAccu
 *
 * \brief Class that represents ViewAccu
 */
struct ViewAccu
{
    int width = 0;

    SERIALIZE_JSON_STRUCT(ViewAccu, width)
};

/*! \class ViewPQ
 *
 * \brief Class that represents ViewPQ
 */
struct View_PQ : public ViewAccu
{
    unsigned start = 0;

    bool operator!=(const View_PQ& rhs) { return ViewAccu::operator!=(rhs) || start != rhs.start; }

    SERIALIZE_JSON_STRUCT(View_PQ, width, start)
};

/*! \class ViewXY
 *
 * \brief Class that represents ViewXY
 */
struct View_XY : public ViewAccu
{
    unsigned start = 0;

    bool operator!=(const View_XY& rhs) { return ViewAccu::operator!=(rhs) || start != rhs.start; }

    SERIALIZE_JSON_STRUCT(View_XY, width, start)
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
    float scale = 0.5f;

    SERIALIZE_JSON_STRUCT(Reticle, display_enabled, scale);

    void Update();
    void Load();
};

/*! \class View
 *
 * \brief Class that represents the view cache
 */
struct Views
{
    ImgType image_type = ImgType::Modulus;
    bool fft_shift = false;
    View_Accu_XY x;
    View_Accu_XY y;
    View_Accu_PQ z;
    View_Accu_PQ z2;
    Windows window;
    bool renorm = false;
    Reticle reticle;

    SERIALIZE_JSON_STRUCT(Views, image_type, fft_shift, x, y, z, z2, window, renorm, reticle);

    void Update();
    void Load();
};

} // namespace holovibes
