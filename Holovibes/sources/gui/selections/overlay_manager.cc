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

#pragma once

#include "overlay_manager.hh"
#include "autofocus_overlay.hh"
#include "zoom_overlay.hh"
#include "noise_overlay.hh"
#include "signal_overlay.hh"
#include "cross_overlay.hh"
#include "filter2d_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		OverlayManager::OverlayManager(BasicOpenGLWindow* parent)
			: current_overlay_(nullptr)
			, parent_(parent)
		{
		}

		OverlayManager::~OverlayManager()
		{}

		void OverlayManager::create_overlay(std::shared_ptr<Overlay> new_overlay)
		{
			overlays_.push_back(new_overlay);
			current_overlay_ = new_overlay;
			current_overlay_->initProgram();
		}

		void OverlayManager::create_autofocus()
		{
			if (current_overlay_)
				current_overlay_->disable();
			create_overlay(std::make_shared<AutofocusOverlay>(parent_));
		}

		void OverlayManager::create_zoom()
		{
			create_overlay(std::make_shared<ZoomOverlay>(parent_));
		}

		void OverlayManager::create_filter2D()
		{
			if (current_overlay_)
				current_overlay_->disable();
			create_overlay(std::make_shared<Filter2DOverlay>(parent_));
		}

		void OverlayManager::create_noise()
		{
			if (current_overlay_ && current_overlay_->getKind() != Signal)
				current_overlay_->disable();
			disable_all(Noise);
			create_overlay(std::make_shared<NoiseOverlay>(parent_));
		}

		void OverlayManager::create_signal()
		{
			if (current_overlay_ && current_overlay_->getKind() != Noise)
				current_overlay_->disable();
			disable_all(Signal);
			create_overlay(std::make_shared<SignalOverlay>(parent_));
		}

		void OverlayManager::create_cross()
		{
			if (current_overlay_)
			current_overlay_->disable();
			create_overlay(std::make_shared<CrossOverlay>(parent_));
		}

		void OverlayManager::press(QPoint pos)
		{
			if (current_overlay_)
				current_overlay_->press(pos);
		}

		void OverlayManager::move(QPoint pos, QSize size)
		{
			if (current_overlay_)
				current_overlay_->move(pos, size);
		}

		void OverlayManager::release(ushort frame)
		{
			if (current_overlay_)
			{
				current_overlay_->release(frame);
				if (!current_overlay_->isActive())
					create_default();
			}
		}

		void OverlayManager::disable_all(KindOfOverlay ko)
		{
			for (auto o : overlays_)
				if (o->getKind() == ko)
					o->disable();
		}

		void OverlayManager::draw()
		{
			//if (overlays_.size() > 1)
				
			for (auto o : overlays_) {
				if (o->isActive() && o->isDisplayed()) {
					o->print();
					o->draw();
				}
			}
		}

		void OverlayManager::clean()
		{
			// Delete all disabled overlays
			overlays_.erase(
				std::remove_if(
					overlays_.begin(),
					overlays_.end(),
					[](std::shared_ptr<Overlay> overlay) { return !overlay->isActive(); }),
				overlays_.end());
			if (overlays_.empty())
				create_default();
		}

		void OverlayManager::reset()
		{
			for (auto o : overlays_)
				o->disable();
			create_default();
		}

		void OverlayManager::create_default()
		{
			switch (parent_->getKindOfView())
			{
			case Direct:
			case Hologram:
				create_zoom();
				break;
			default:
				// TODO: Single cross
				current_overlay_.reset();
				break;
			}
		}

		const Rectangle& OverlayManager::getZone() const
		{
			return current_overlay_->getZone();
		}

		KindOfOverlay OverlayManager::getKind() const
		{
			return current_overlay_ ? current_overlay_->getKind() : Zoom;
		}

		bool OverlayManager::setCrossBuffer(QPoint pos, QSize frame)
		{
			return false;
		}

		bool OverlayManager::setDoubleCrossBuffer(QPoint pos, QPoint pos2, QSize frame)
		{
			return false;
		}

		void OverlayManager::printVector()
		{
			std::cout << std::endl;
			for (auto o : overlays_)
				o->print();
			std::cout << std::endl;
		}
	}
}
