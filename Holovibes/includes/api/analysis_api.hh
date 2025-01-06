/*! \file analysis_api.hh
 *
 * \brief Regroup all functions used to interact with analysis settings.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{
class AnalysisApi : public IApi
{

  public:
    AnalysisApi(const Api* api)
        : IApi(api)
    {
    }
    /*!
     * \name Artery Mask
     *
     */
    inline bool get_artery_mask_enabled() { return GET_SETTING(ArteryMaskEnabled); }
    void set_artery_mask_enabled(bool value);

    /*!
     * \name Vein Mask
     *
     */
    inline bool get_vein_mask_enabled() { return GET_SETTING(VeinMaskEnabled); }
    void set_vein_mask_enabled(bool value);

    /*!
     * \name Choroid Mask
     *
     */
    inline bool get_choroid_mask_enabled() { return GET_SETTING(ChoroidMaskEnabled); }
    void set_choroid_mask_enabled(bool value);

    /*!
     * \brief Time Window
     *
     */

    inline int get_time_window() { return GET_SETTING(TimeWindow); }
    void set_time_window(int value);

    /*!
     * \name Vesselness Sigma
     *
     */
    inline double get_vesselness_sigma() { return GET_SETTING(VesselnessSigma); }
    void set_vesselness_sigma(double value);

    inline int get_min_mask_area() { return GET_SETTING(MinMaskArea); }
    inline void set_min_mask_area(int value) { return UPDATE_SETTING(MinMaskArea, value); }

    inline float get_diaphragm_factor() { return GET_SETTING(DiaphragmFactor); }
    inline void set_diaphragm_factor(float value)
    {
        value = value > 1.0f ? 1.0f : (value < 0 ? 0 : value);
        return UPDATE_SETTING(DiaphragmFactor, value);
    }
    inline bool get_diaphragm_preview_enabled() { return GET_SETTING(DiaphragmPreviewEnabled); }
    inline void set_diaphragm_preview_enabled(bool value) { return UPDATE_SETTING(DiaphragmPreviewEnabled, value); }

    inline float get_barycenter_factor() { return GET_SETTING(BarycenterFactor); }
    inline void set_barycenter_factor(float value)
    {
        value = value > 1.0f ? 1.0f : (value < 0 ? 0 : value);
        return UPDATE_SETTING(BarycenterFactor, value);
    }
    inline bool get_barycenter_preview_enabled() { return GET_SETTING(BarycenterPreviewEnabled); }
    inline void set_barycenter_preview_enabled(bool value) { return UPDATE_SETTING(BarycenterPreviewEnabled, value); }
};
} // namespace holovibes::api