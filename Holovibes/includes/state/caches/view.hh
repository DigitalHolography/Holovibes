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

//! \brief Type of the image displayed
using ImgTypeParam = CustomParameter<ImgType, ImgType::Modulus, "img_type", ImgType>;
using ViewAccuX = CustomParameter<View_XY, View_XY{}, "view_accu_x">;
using ViewAccuY = CustomParameter<View_XY, View_XY{}, "view_accu_y">;
using ViewAccuP = CustomParameter<View_PQ, View_PQ{}, "view_accu_p">;
using ViewAccuQ = CustomParameter<View_PQ, View_PQ{}, "view_accu_q">;
using ViewXY = CustomParameter<View_XYZ, View_XYZ{}, "view_xy">;
using ViewXZ = CustomParameter<View_XYZ, View_XYZ{}, "view_xz">;
using ViewYZ = CustomParameter<View_XYZ, View_XYZ{}, "view_yz">;
using Filter2D = CustomParameter<View_Window, View_Window{}, "filter2d">;
using CurrentWindowKind = CustomParameter<WindowKind, WindowKind::XYview, "current_window">;
using LensViewEnabled = BoolParameter<false, "lens_view_enabled">;
//! \brief Enables the signal and noise chart display
using ChartDisplayEnabled = BoolParameter<false, "chart_display_enabled">;
//! \brief Enables filter 2D
using Filter2DEnabled = BoolParameter<false, "filter2d_enabled">;
//! \brief Enables filter 2D View
using Filter2DViewEnabled = BoolParameter<false, "filter2d_view_enabled">;
//! \brief Is shift fft enabled (switching representation diagram) --- check for timetranformation size
using FftShiftEnabled = BoolParameter<false, "fft_shift_enabled">;
//! \brief Display the raw interferogram when we are in hologram mode.
using RawViewEnabled = BoolParameter<false, "raw_view_enabled">;
//! \brief Are slices YZ and XZ enabled
using CutsViewEnabled = BoolParameter<false, "cuts_view_enabled">;
using RenormEnabled = BoolParameter<true, "renorm_enabled">;
//! \brief Is the reticle overlay enabled
using ReticleDisplayEnabled = BoolParameter<false, "reticle_display_enabled">;
//! \brief Reticle border scale
using ReticleScale = FloatParameter<0.5f, "reticle_scale">;

using ViewCache = MicroCache<ImgTypeParam,
                             ViewAccuX,
                             ViewAccuY,
                             ViewAccuP,
                             ViewAccuQ,
                             ViewXY,
                             ViewXZ,
                             ViewYZ,
                             Filter2D,
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
                             ReticleScale>;

} // namespace holovibes
