/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "composite_area_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include <iostream>

using holovibes::gui::CompositeAreaOverlay;
using holovibes::gui::BasicOpenGLWindow;


CompositeAreaOverlay::CompositeAreaOverlay(BasicOpenGLWindow* parent)
	: RectOverlay(KindOfOverlay::Stabilization, parent)
{
	color_ = { 0.6f, 0.5f, 0.0f };
}

void CompositeAreaOverlay::release(ushort frameSide)
{
	disable();

	if (zone_.topLeft() == zone_.bottomRight())
		return;

	parent_->getCd()->setCompositeZone(zone_);
}


