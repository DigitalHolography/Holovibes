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

#include "strip_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		StripOverlay::StripOverlay(BasicOpenGLWindow* parent,
			Component& component,
			std::atomic<ushort>& nsamples,
			Color color)
			: RectOverlay(KindOfOverlay::Strip, parent)
			, component_(component)
			, nsamples_(nsamples)
		{
			color_ = color;
			alpha_ = 0.4f;
			display_ = true;
		}

		void StripOverlay::draw()
		{
			compute_zone();
			setBuffer();
			RectOverlay::draw();
		}

		void StripOverlay::compute_zone()
		{
			ushort pmin = component_.p_min.load();
			ushort pmax = component_.p_max.load() + 1;
			if (parent_->getKindOfView() == SliceXZ)
			{
				float ratio = parent_->height() / (nsamples_.load() - 1);
				QPoint topleft(0, ratio * pmin);
				QPoint bottomRight(parent_->width(), ratio * pmax);
				zone_ = QRect(topleft, bottomRight);
			}
			else
			{
				float ratio = parent_->width() / (nsamples_.load() - 1);
				QPoint topleft(ratio * pmin, 0);
				QPoint bottomRight(ratio * pmax, parent_->height());
				zone_ = QRect(topleft, bottomRight);
			}
		}
	}
}
