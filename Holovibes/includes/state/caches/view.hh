/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

#include "view_struct.hh"
#include "enum_img_type.hh"
#include "enum_window_kind.hh"

namespace holovibes
{

// clang-format off

//! \brief Type of the image displayed
class ImageType_PARAM : public CustomParameter<ImageTypeEnum, ImageTypeEnum::Modulus, "img_type", ImageTypeEnum>{};
class ViewAccuX : public CustomParameter<View_XY, DefaultLiteral<View_XY>{}, "view_accu_x">{};
class ViewAccuY : public CustomParameter<View_XY, DefaultLiteral<View_XY>{}, "view_accu_y">{};
class ViewAccuP : public CustomParameter<View_PQ, DefaultLiteral<View_PQ>{}, "view_accu_p">{};
class ViewAccuQ : public CustomParameter<View_PQ, DefaultLiteral<View_PQ>{}, "view_accu_q">{};
class ViewXY_PARAM : public CustomParameter<View_XYZ, DefaultLiteral<View_XYZ>{}, "view_xy">{};
class ViewXZ_PARAM : public CustomParameter<View_XYZ, DefaultLiteral<View_XYZ>{}, "view_xz">{};
class ViewYZ_PARAM : public CustomParameter<View_XYZ, DefaultLiteral<View_XYZ>{}, "view_yz">{};
class Filter2D_PARAM : public CustomParameter<View_Window, DefaultLiteral<View_Window>{}, "filter2d">{};
class CurrentWindowKind : public CustomParameter<WindowKind, WindowKind::XYview, "current_window">{};
class LensViewEnabled : public BoolParameter<false, "lens_view_enabled">{};
//! \brief Enables the signal and noise chart display
class ChartDisplayEnabled : public BoolParameter<false, "chart_display_enabled">{};
//! \brief Enables filter 2D
class Filter2DEnabled : public BoolParameter<false, "filter2d_enabled">{};
//! \brief Enables filter 2D View
class Filter2DViewEnabled : public BoolParameter<false, "filter2d_view_enabled">{};
//! \brief Is shift fft enabled (switching representation diagram) --- check for timetranformation size
class FftShiftEnabled : public BoolParameter<false, "fft_shift_enabled">{};
//! \brief Display the raw interferogram when we are in hologram mode.
class RawViewEnabled : public BoolParameter<false, "raw_view_enabled">{};
//! \brief Are slices YZ and XZ enabled
class CutsViewEnabled : public BoolParameter<false, "cuts_view_enabled">{};
class RenormEnabled : public BoolParameter<true, "renorm_enabled">{};
//! \brief Is the reticle overlay enabled
class ReticleDisplayEnabled : public BoolParameter<false, "reticle_display_enabled">{};
//! \brief Reticle border scale
class ReticleScale : public FloatParameter<0.5f, "reticle_scale">{};

// clang-format on

class ViewCache : public MicroCache<ImageType_PARAM,
                                    ViewAccuX,
                                    ViewAccuY,
                                    ViewAccuP,
                                    ViewAccuQ,
                                    ViewXY_PARAM,
                                    ViewXZ_PARAM,
                                    ViewYZ_PARAM,
                                    Filter2D_PARAM,
                                    CurrentWindowKind,
                                    LensViewEnabled,
                                    ChartDisplayEnabled,
                                    Filter2DEnabled,
                                    Filter2DViewEnabled,
                                    FftShiftEnabled,
                                    RawViewEnabled,
                                    CutsViewEnabled,
                                    RenormEnabled,
                                    ReticleDisplayEnabled,
                                    ReticleScale>
{
};

} // namespace holovibes
