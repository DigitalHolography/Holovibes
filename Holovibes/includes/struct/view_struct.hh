/*! \file
 *
 * \brief View structure
 *
 */

#pragma once

#include "logger.hh"
#include "all_struct.hh"
#include "enum_img_type.hh"

typedef unsigned int uint;

namespace holovibes
{
/*! \class ViewContrast
 *
 * \brief Class that represents ViewContrast
 */
struct ViewContrast
{
    bool enabled = true;
    bool auto_refresh = true;
    bool invert = false;
    float min = 1.f;
    float max = 65535.f;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(ViewContrast, enabled, auto_refresh, invert, min, max);

    bool operator==(const ViewContrast& other) const
    {
        return (enabled == other.enabled && auto_refresh == other.auto_refresh && invert == other.invert &&
                min == other.min && max == other.max);
    }
};

/*! \class ViewWindow
 *
 * \brief Class that represents ViewWindow
 */
struct ViewWindow
{
    bool log_enabled = false;

    ViewContrast contrast;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(ViewWindow, log_enabled, contrast);

    bool operator==(const ViewWindow& other) const
    {
        return (log_enabled == other.log_enabled && contrast == other.contrast);
    }
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
    bool enabled = false;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(ViewXYZ, log_enabled, contrast, horizontal_flip, rotation, output_image_accumulation, enabled);

    bool operator==(const ViewXYZ& other) const
    {
        return (horizontal_flip == other.horizontal_flip && rotation == other.rotation &&
                output_image_accumulation == other.output_image_accumulation && log_enabled == other.log_enabled &&
                contrast == other.contrast);
    }
};

/*! \class ViewAccu
 *
 * \brief Class that represents ViewAccu
 */
struct ViewAccu
{
    int width = 0;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(ViewAccu, width);
};

/*! \class ViewPQ
 *
 * \brief Class that represents ViewPQ
 */
struct ViewPQ : public ViewAccu
{
    unsigned start = 0;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(ViewPQ, width, start);

    bool operator==(const ViewPQ& other) const { return (start == other.start && width == other.width); }
};

/*! \class ViewXY
 *
 * \brief Class that represents ViewXY
 */
struct ViewXY : public ViewAccu
{
    unsigned start = 0;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(ViewXY, width, start);

    bool operator==(const ViewXY& other) const { return (start == other.start && width == other.width); }
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

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(Windows, xy, yz, xz, filter2d);

    /*!
     * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
     * synchronize variables of Windows with the one in GSH, update variables of GSH
     * with the one of Windows and assert that the Windows variables are valid
     */
    SETTING_RELATED_FUNCTIONS();
};

/*! \class Reticle
 *
 * \brief Class that represents the Reticle
 */
struct Reticle
{
    bool display_enabled = false;
    float scale = 0.5f;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(Reticle, display_enabled, scale);

    /*!
     * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
     * synchronize variables of Reticle with the one in GSH, update variables of GSH
     * with the one of Reticle and assert that the Reticle variables are valid
     */
    SETTING_RELATED_FUNCTIONS();
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

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(Views, image_type, fft_shift, x, y, z, z2, window, renorm, reticle);

    /*!
     * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
     * synchronize variables of Views with the one in GSH, update variables of GSH
     * with the one of Views and assert that the Views variables are valid
     */
    SETTING_RELATED_FUNCTIONS();
};

} // namespace holovibes
