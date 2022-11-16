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
    bool horizontal_flip = false;
    float rotation = 0;
    unsigned output_image_accumulation = 1;

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
struct ViewPQ : public ViewAccu
{
    unsigned start = 0;

    SERIALIZE_JSON_STRUCT(ViewPQ, width, start)
};

/*! \class ViewXY
 *
 * \brief Class that represents ViewXY
 */
struct ViewXY : public ViewAccu
{
    unsigned start = 0;

    SERIALIZE_JSON_STRUCT(ViewXY, width, start)
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
    ViewXY x;
    ViewXY y;
    ViewPQ z;
    ViewPQ z2;
    Windows window;
    bool renorm = false;
    Reticle reticle;

    SERIALIZE_JSON_STRUCT(Views, image_type, fft_shift, x, y, z, z2, window, renorm, reticle);

    void Update();
    void Load();
};

} // namespace holovibes
