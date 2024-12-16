/*! \file
 *
 * \brief Enum for the types of curves displayable in the analysis panel.
 */
#pragma once

namespace holovibes::gui
{
/*! \enum AnalysisCurveName
 *
 *  \brief The different types of displayable curves for the chart graph
 *  of the analysis panel.
 *
 */
enum AnalysisCurveName
{
    ARTERY_MEAN = 0,
    VEIN_MEAN = 1,
    CHOROID_MEAN = 2,
    ALL_MEAN = 3
};
} // namespace holovibes::gui