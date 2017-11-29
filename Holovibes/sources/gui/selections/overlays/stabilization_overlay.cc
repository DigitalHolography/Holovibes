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

#include "stabilization_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "power_of_two.hh"

using holovibes::gui::StabilizationOverlay;
using holovibes::gui::BasicOpenGLWindow;


StabilizationOverlay::StabilizationOverlay(BasicOpenGLWindow* parent)
	: RectOverlay(KindOfOverlay::Stabilization, parent)
{
	color_ = { 0.8f, 0.4f, 0.4f };
	parent_->getCd()->xy_stabilization_paused = true;
}

void StabilizationOverlay::release(ushort frameSide)
{
	disable();

	if (zone_.topLeft() == zone_.bottomRight())
		return;

	make_pow2_square();

	parent_->getCd()->setStabilizationZone(zone_);
	parent_->getCd()->xy_stabilization_paused = false;
}

void StabilizationOverlay::make_pow2_square()
{
	zone_.setX(std::max(zone_.x().get(), 0));
	zone_.setY(std::max(zone_.y().get(), 0));
	zone_.setBottom(std::min(zone_.bottom().get(), static_cast<int>(parent_->getFd().height)));
	zone_.setRight(std::min(zone_.right().get(), static_cast<int>(parent_->getFd().width)));

	const int min = prevPowerOf2(1 + std::min(std::abs(zone_.width()), std::abs(zone_.height())));
	zone_.setDst(units::PointFd(units::ConversionData(parent_),
		zone_.src().x() + ((zone_.src().x() < zone_.dst().x()) ? min : -min),
		zone_.src().y() + ((zone_.src().y() < zone_.dst().y()) ? min : -min)
	));
}

void StabilizationOverlay::move(QMouseEvent* e)
{
	if (e->buttons() == Qt::LeftButton)
	{
		display_ = true;
		zone_.setDst(getMousePos(e->pos()));
		make_pow2_square();
		setBuffer();
	}
}

