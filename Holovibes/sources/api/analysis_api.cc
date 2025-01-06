#include "analysis_api.hh"

#include "API.hh"

namespace holovibes::api
{

void AnalysisApi::set_artery_mask_enabled(bool value)
{
    UPDATE_SETTING(ArteryMaskEnabled, value);
    api_->compute.pipe_refresh();
}

/*!
 * \name Vein Mask
 *
 */
void AnalysisApi::set_vein_mask_enabled(bool value)
{
    UPDATE_SETTING(VeinMaskEnabled, value);
    api_->compute.pipe_refresh();
}

/*!
 * \name Choroid Mask
 *
 */
void AnalysisApi::set_choroid_mask_enabled(bool value)
{
    UPDATE_SETTING(ChoroidMaskEnabled, value);
    api_->compute.pipe_refresh();
}

/*!
 * \brief Time Window
 *
 */

void AnalysisApi::set_time_window(int value)
{
    UPDATE_SETTING(TimeWindow, value);
    api_->compute.pipe_refresh();
}

/*!
 * \name Vesselness Sigma
 *
 */
void AnalysisApi::set_vesselness_sigma(double value)
{
    UPDATE_SETTING(VesselnessSigma, value);
    api_->compute.pipe_refresh();
}

} // namespace holovibes::api