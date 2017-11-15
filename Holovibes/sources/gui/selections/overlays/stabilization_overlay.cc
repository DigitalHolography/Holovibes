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

	zone_.setX(std::max(zone_.x().get(), 0));
	zone_.setY(std::max(zone_.y().get(), 0));
	zone_.setBottom(std::min(zone_.bottom().get(), parent_->height()));
	zone_.setRight(std::min(zone_.right().get(), parent_->width()));

	units::RectFd texZone = zone_;

	const uint square_size = prevPowerOf2(std::min(texZone.width().get(), texZone.height().get()));
	texZone.setWidth(square_size);
	texZone.setHeight(square_size);

	parent_->getCd()->setStabilizationZone(texZone);
	parent_->getCd()->xy_stabilization_paused = false;
}

void StabilizationOverlay::make_pow2_square()
{
	units::RectFd fd_zone = zone_;
	const int min = prevPowerOf2(std::min(std::abs(fd_zone.width()), std::abs(fd_zone.height())));
	units::PointFd bottom_right = fd_zone.topLeft();
	bottom_right.x() += min;
	bottom_right.y() += min;
	fd_zone.setBottomRight(bottom_right);
	zone_ = fd_zone;
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

