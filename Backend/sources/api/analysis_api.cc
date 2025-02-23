#include "analysis_api.hh"

#include "API.hh"

namespace holovibes::api
{

void AnalysisApi::set_artery_mask_enabled(bool value) { UPDATE_SETTING(ArteryMaskEnabled, value); }

/*!
 * \name Vein Mask
 *
 */
void AnalysisApi::set_vein_mask_enabled(bool value) { UPDATE_SETTING(VeinMaskEnabled, value); }

/*!
 * \name Choroid Mask
 *
 */
void AnalysisApi::set_choroid_mask_enabled(bool value) { UPDATE_SETTING(ChoroidMaskEnabled, value); }

/*!
 * \brief Time Window
 *
 */

void AnalysisApi::set_time_window(int value) { UPDATE_SETTING(TimeWindow, value); }

/*!
 * \name Vesselness Sigma
 *
 */
void AnalysisApi::set_vesselness_sigma(double value) { UPDATE_SETTING(VesselnessSigma, value); }

/*!
 * \name Threshold
 *
 */
void AnalysisApi::set_threshold(float value) { UPDATE_SETTING(Threshold, value); }

} // namespace holovibes::api