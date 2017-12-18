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

#include "composite_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		CompositeOverlay::CompositeOverlay(BasicOpenGLWindow* parent)
			: Overlay(KindOfOverlay::Composite, parent)
		{
		}

		void CompositeOverlay::release(ushort frameSide)
		{
			is_active_ = false;
		}
		void CompositeOverlay::move(QMouseEvent *e)
		{
			if (is_active_)
			{
				zone_.setDst(getMousePos(e->pos()));
				parent_->getCd()->component_r.p_min = check_interval(zone_.src().x());
				parent_->getCd()->component_b.p_max = check_interval(zone_.dst().x());
				parent_->getCd()->notify_observers();
			}
		}
		void CompositeOverlay::press(QMouseEvent* e)
		{
			Overlay::press(e);
			is_active_ = true;
		}

		int CompositeOverlay::check_interval(int x)
		{
			const int max = parent_->getCd()->nsamples - 1;
			return std::min(max, std::max(x, 0));
		}
	}
}
