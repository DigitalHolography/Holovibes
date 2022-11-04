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

    bool operator!=(const ViewContrast& rhs)
    {
        return enabled != rhs.enabled || auto_refresh != rhs.auto_refresh || invert != rhs.invert || min != rhs.min ||
               max != rhs.max;
    }

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
    bool get_exec_auto_contrast() const { return exec_auto_contrast_; }
    void request_exec_auto_contrast() { exec_auto_contrast_ = true; }
    void reset_exec_auto_contrast() { exec_auto_contrast_ = false; }

    bool operator!=(const ViewWindow& rhs) { return contrast != rhs.contrast || log_enabled != rhs.log_enabled; }

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
    uint image_accumulation_level = 1;

    bool is_image_accumulation_enabled() const { return image_accumulation_level > 1; }

    // FIXME : CHEC THIS TO TRIGGER
  private:
    bool request_clear_image_accumulation_ = false;

  public:
    bool get_request_clear_image_accumulation() const { return request_clear_image_accumulation_; }
    void request_clear_image_accumulation() { request_clear_image_accumulation_ = true; }
    void reset_request_clear_image_accumulation() { request_clear_image_accumulation_ = false; }

  public:
    bool operator!=(const ViewXYZ& rhs)
    {
        return ViewWindow::operator!=(rhs) || log_enabled != rhs.log_enabled || flip_enabled != rhs.flip_enabled ||
               rotation != rhs.rotation || image_accumulation_level != rhs.image_accumulation_level;
     }

    SERIALIZE_JSON_STRUCT(ViewXYZ, log_enabled, contrast, flip_enabled, rotation, image_accumulation_level)
};

/*! \class ViewAccu
 *
 * \brief Class that represents ViewAccu
 */
struct ViewAccu
{
    int accu_level = 0;

    bool operator!=(const ViewAccu& rhs) { return accu_level != rhs.accu_level; }

    SERIALIZE_JSON_STRUCT(ViewAccu, accu_level)
};

/*! \class ViewPQ
 *
 * \brief Class that represents ViewPQ
 */
struct ViewAccuPQ : public ViewAccu
{
    unsigned index = 0;

    bool operator!=(const ViewAccuPQ& rhs) { return ViewAccu::operator!=(rhs) || index != rhs.index; }

    SERIALIZE_JSON_STRUCT(ViewAccuPQ, accu_level, index)
};

/*! \class ViewXY
 *
 * \brief Class that represents ViewXY
 */
struct ViewAccuXY : public ViewAccu
{
    unsigned cuts = 0;

    bool operator!=(const ViewAccuXY& rhs) { return ViewAccu::operator!=(rhs) || cuts != rhs.cuts; }

    SERIALIZE_JSON_STRUCT(ViewAccuXY, accu_level, cuts)
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
struct ReticleStruct
{
    bool display_enabled = false;
    float reticle_scale = 0.5f;

    SERIALIZE_JSON_STRUCT(ReticleStruct, display_enabled, reticle_scale);
};

/*! \class View
 *
 * \brief Class that represents the view cache
 */
struct Views
{
    ImageTypeEnum img_type = ImageTypeEnum::Modulus;
    bool fft_shift = false;
    ViewAccuXY x;
    ViewAccuXY y;
    ViewAccuPQ p;
    ViewAccuPQ q;
    Windows window;
    bool renorm = false;
    ReticleStruct reticle;

    SERIALIZE_JSON_STRUCT(Views, img_type, fft_shift, x, y, p, q, window, renorm, reticle);

    void Update();
    void Load();
};

} // namespace holovibes
