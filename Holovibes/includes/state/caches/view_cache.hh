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
class ImageType : public CustomParameter<ImageTypeEnum, ImageTypeEnum::Modulus, "img_type", ImageTypeEnum>{};
class ViewAccuX : public CustomParameter<ViewAccuXY, DefaultLiteral<ViewAccuXY>{}, "view_accu_x">{};
class ViewAccuY : public CustomParameter<ViewAccuXY, DefaultLiteral<ViewAccuXY>{}, "view_accu_y">{};
class ViewAccuP : public CustomParameter<ViewAccuPQ, DefaultLiteral<ViewAccuPQ>{}, "view_accu_p">{};
class ViewAccuQ : public CustomParameter<ViewAccuPQ, DefaultLiteral<ViewAccuPQ>{}, "view_accu_q">{};
class ViewXY : public CustomParameter<ViewXYZ, DefaultLiteral<ViewXYZ>{}, "view_xy">{};
class ViewXZ : public CustomParameter<ViewXYZ, DefaultLiteral<ViewXYZ>{}, "view_xz">{};
class ViewYZ : public CustomParameter<ViewXYZ, DefaultLiteral<ViewXYZ>{}, "view_yz">{};
class ViewFilter2D : public CustomParameter<ViewWindow, DefaultLiteral<ViewWindow>{}, "view_filter2d">{};
class CurrentWindowKind : public CustomParameter<WindowKind, WindowKind::XYview, "current_window">{};
class LensViewEnabled : public BoolParameter<false, "lens_view_enabled">{};
//! \brief Enables the signal and noise chart display
class ChartDisplayEnabled : public BoolParameter<false, "chart_display_enabled">{};
//! \brief Enables filter 2D View
class Filter2DViewEnabled : public BoolParameter<false, "filter2d_view_enabled">{};
//! \brief Is shift fft enabled (switching representation diagram) --- check for timetranformation size
class FftShiftEnabled : public BoolParameter<false, "fft_shift_enabled">{};
//! \brief Display the raw interferogram when we are in hologram mode.
class RawViewEnabled : public BoolParameter<false, "raw_view_enabled">{};
//! \brief Are slices YZ and XZ enabled
class CutsViewEnabled : public BoolParameter<false, "cuts_view_enabled">{};
class RenormEnabled : public BoolParameter<true, "renorm_enabled">{};
class Reticle : public CustomParameter<ReticleStruct, DefaultLiteral<ReticleStruct>{}, "reticle">{};

// clang-format on

class ViewCache : public MicroCache<ImageType,
                                    ViewAccuX,
                                    ViewAccuY,
                                    ViewAccuP,
                                    ViewAccuQ,
                                    ViewXY,
                                    ViewXZ,
                                    ViewYZ,
                                    ViewFilter2D,
                                    CurrentWindowKind,
                                    LensViewEnabled,
                                    ChartDisplayEnabled,
                                    Filter2DViewEnabled,
                                    FftShiftEnabled,
                                    RawViewEnabled,
                                    CutsViewEnabled,
                                    RenormEnabled,
                                    Reticle>
{
};

} // namespace holovibes
