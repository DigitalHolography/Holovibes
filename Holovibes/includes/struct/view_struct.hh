/*! \file
 *
 * \brief View structure
 *
 */

#pragma once

#include "types.hh"
#include "logger.hh"
#include "all_struct.hh"
#include "enum_img_type.hh"

#define CONSTRUCTOR(name, arg_name)

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

    bool operator!=(const ViewContrast& rhs) const
    {
        return enabled != rhs.enabled || auto_refresh != rhs.auto_refresh || invert != rhs.invert || min != rhs.min ||
               max != rhs.max;
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

    SERIALIZE_JSON_STRUCT(ViewWindow, log_enabled, contrast)

    bool operator!=(const ViewWindow& rhs) const { return contrast != rhs.contrast || log_enabled != rhs.log_enabled; }

  public:
    float get_contrast_min_logged() const
    {
        if (log_enabled)
            return contrast.min;
        return log10(contrast.min);
    }

    float get_contrast_max_logged() const
    {
        if (log_enabled)
            return contrast.max;
        return log10(contrast.max);
    }
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
    uint output_image_accumulation = 1;

    bool is_image_accumulation_enabled() const { return output_image_accumulation > 1; }

  public:
    SERIALIZE_JSON_STRUCT(ViewXYZ, log_enabled, contrast, horizontal_flip, rotation, output_image_accumulation)

    bool operator!=(const ViewXYZ& rhs) const
    {
        return ViewWindow::operator!=(rhs) || log_enabled != rhs.log_enabled || horizontal_flip != rhs.horizontal_flip ||
               rotation != rhs.rotation || output_image_accumulation != rhs.output_image_accumulation;
    }
};

/*! \class ViewAccu
 *
 * \brief Class that represents ViewAccu
 */
struct ViewAccu
{
    int width = 0;

    SERIALIZE_JSON_STRUCT(ViewAccu, width)

    bool operator!=(const ViewAccu& rhs) const { return width != rhs.width; }
};

/*! \class ViewPQ
 *
 * \brief Class that represents ViewPQ
 */
struct ViewAccuPQ : public ViewAccu
{
    unsigned start = 0;

    SERIALIZE_JSON_STRUCT(ViewAccuPQ, width, start)

    bool operator!=(const ViewAccuPQ& rhs) const { return ViewAccu::operator!=(rhs) || start != rhs.start; }
};

/*! \class ViewXY
 *
 * \brief Class that represents ViewXY
 */
struct ViewAccuXY : public ViewAccu
{
    unsigned start = 0;

    SERIALIZE_JSON_STRUCT(ViewAccuXY, width, start)

    bool operator!=(const ViewAccuXY& rhs) const { return ViewAccu::operator!=(rhs) || start != rhs.start; }
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

/*! \class ReticleStruct
 *
 * \brief Class that represents the ReticleStruct
 */
struct ReticleStruct
{
    bool display_enabled = false;
    float scale = 0.5f;

    SERIALIZE_JSON_STRUCT(ReticleStruct, display_enabled, scale);

    bool operator!=(const ReticleStruct& rhs) const
    {
        return display_enabled != rhs.display_enabled || scale != rhs.scale;
    }
};

/*! \class View
 *
 * \brief Class that represents the view cache
 */
struct Views
{
    ImageTypeEnum image_type = ImageTypeEnum::Modulus;
    bool fft_shift = false;
    ViewAccuXY x;
    ViewAccuXY y;
    ViewAccuPQ z;
    ViewAccuPQ z2;
    Windows window;
    bool renorm = false;
    ReticleStruct reticle;

    SERIALIZE_JSON_STRUCT(Views, image_type, fft_shift, x, y, z, z2, window, renorm, reticle);

    void Update();
    void Load();
};

inline std::ostream& operator<<(std::ostream& os, const ViewContrast& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ViewWindow& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ViewXYZ& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ViewAccu& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ViewAccuPQ& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ViewAccuXY& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const Windows& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ReticleStruct& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const Views& value) { return os << json{value}; }
} // namespace holovibes
