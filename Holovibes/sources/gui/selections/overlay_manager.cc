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
#include "zoom_overlay.hh"
#include "noise_overlay.hh"
#include "signal_overlay.hh"
#include "cross_overlay.hh"
#include "slice_cross_overlay.hh"
#include "filter2d_overlay.hh"
#include "filter2d_subzone_overlay.hh"
#include "composite_area_overlay.hh"
#include "rainbow_overlay.hh"
#include "reticle_overlay.hh"

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

		template <KindOfOverlay ko>
		void OverlayManager::create_overlay()
		{
			return;
		}

		template <>
		void OverlayManager::create_overlay<Zoom>()
		{
			if (!set_current(Zoom))
				create_overlay(std::make_shared<ZoomOverlay>(parent_));
		}

		template <>
		void OverlayManager::create_overlay<Filter2D>()
		{
			if (!set_current(Filter2D))
				create_overlay(std::make_shared<Filter2DOverlay>(parent_));
		}

		template<>
		void OverlayManager::create_overlay<Filter2DSubZone>()
		{
			if (!set_current(Filter2DSubZone))
			{
				auto &filter2d_overlay = std::dynamic_pointer_cast<Filter2DOverlay>(current_overlay_);
				create_overlay(std::make_shared<Filter2DSubZoneOverlay>(parent_));
				std::dynamic_pointer_cast<Filter2DSubZoneOverlay>(overlays_.back())->setFilter2dOverlay(filter2d_overlay);
			}
		}

		template <>
		void OverlayManager::create_overlay<Noise>()
		{
			if (!set_current(Noise))
			{
				std::shared_ptr<Overlay> noise_overlay = std::make_shared<NoiseOverlay>(parent_);
				create_overlay(noise_overlay);
				noise_overlay_ = noise_overlay;
			}
		}

		template <>
		void OverlayManager::create_overlay<Signal>()
		{
			if (!set_current(Signal))
			{
				std::shared_ptr<Overlay> signal_overlay = std::make_shared<SignalOverlay>(parent_);
				create_overlay(signal_overlay);
				signal_overlay_ = signal_overlay;
			}
		}

		template <>
		void OverlayManager::create_overlay<Cross>()
		{
			if (!set_current(Cross))
				create_overlay(std::make_shared<CrossOverlay>(parent_));
			create_overlay<Zoom>();
		}

		template <>
		void OverlayManager::create_overlay<SliceCross>()
		{
			if (!set_current(SliceCross))
				create_overlay(std::make_shared<SliceCrossOverlay>(parent_));
		}

		template <>
		void OverlayManager::create_overlay<KindOfOverlay::CompositeArea>()
		{
			if (!set_current(KindOfOverlay::CompositeArea))
				create_overlay(std::make_shared<CompositeAreaOverlay>(parent_));
		}

		template<>
		void OverlayManager::create_overlay<Rainbow>()
		{
			if (!set_current(KindOfOverlay::Rainbow))
				create_overlay(std::make_shared<RainbowOverlay>(parent_));
		}

		template<>
		void OverlayManager::create_overlay<Reticle>()
		{
			if (!set_current(KindOfOverlay::Reticle))
				create_overlay(std::make_shared<ReticleOverlay>(parent_));
		}

		void OverlayManager::create_overlay(std::shared_ptr<Overlay> new_overlay)
		{
			clean();
			overlays_.push_back(new_overlay);
			set_current(new_overlay);
			current_overlay_->initProgram();
		}

		bool OverlayManager::set_current(KindOfOverlay ko)
		{
			for (auto o : overlays_)
				if (o->getKind() == ko && o->isActive())
				{
					set_current(o);
					return true;
				}
			return false;
		}

		void OverlayManager::set_current(std::shared_ptr<Overlay> new_overlay)
		{
			current_overlay_ = new_overlay;
			parent_->getCd()->notify_observers();
		}

		void OverlayManager::press(QMouseEvent *e)
		{
			if (current_overlay_)
				current_overlay_->press(e);
		}

		void OverlayManager::keyPress(QKeyEvent *e)
		{
			// Reserving space for moving the cross
			if (e->key() == Qt::Key_Space)
			{
				for (auto o : overlays_)
					if ((o->getKind() == Cross || o->getKind() == SliceCross) && o->isActive())
						o->keyPress(e);
			}
			else if (current_overlay_)
				current_overlay_->keyPress(e);
		}

		void OverlayManager::move(QMouseEvent *e)
		{
			for (auto o : overlays_)
				if ((o->getKind() == Cross || o->getKind() == SliceCross) && o->isActive())
					o->move(e);
			if (current_overlay_)
				current_overlay_->move(e);
		}

		void OverlayManager::release(ushort frame)
		{
			if (current_overlay_)
			{
				current_overlay_->release(frame);
				if (!current_overlay_->isActive())
					create_default();
				else if (current_overlay_->getKind() == Noise)
					create_overlay<Signal>();
				else if (current_overlay_->getKind() == Signal)
					create_overlay<Noise>();
				else if (current_overlay_->getKind() == Filter2D && parent_->getCd()->filter_2d_type == Filter2DType::BandPass)
					create_overlay<Filter2DSubZone>();
			}
		}

		bool OverlayManager::disable_all(KindOfOverlay ko)
		{
			bool found = false;
			for (auto o : overlays_)
				if (o->getKind() == ko)
				{
					o->disable();
					found = true;
				}
			return found;
		}

		void OverlayManager::draw()
		{
			for (auto o : overlays_)
				if (o->isActive() && o->isDisplayed())
					o->draw();
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
		}

		void OverlayManager::reset(bool def)
		{
			for (auto o : overlays_)
				o->disable();
			if (def)
				create_default();
		}

		void OverlayManager::create_default()
		{
			switch (parent_->getKindOfView())
			{
			case Raw:
			case Hologram:
				create_overlay<Zoom>();
				break;
			case SliceXZ:
			case SliceYZ:
				create_overlay<SliceCross>();
				break;
			default:
				break;
			}
		}

		units::RectWindow OverlayManager::getZone() const
		{
			assert(current_overlay_);
			return current_overlay_->getZone();
		}

		KindOfOverlay OverlayManager::getKind() const
		{
			return current_overlay_ ? current_overlay_->getKind() : Zoom;
		}

		# ifdef _DEBUG
		void OverlayManager::printVector()
		{
			std::cout << std::endl;
			std::cout << "Current overlay :" << std::endl;
			if (current_overlay_)
				current_overlay_->print();
			std::cout << std::endl;
			for (auto o : overlays_)
				o->print();
			std::cout << std::endl;
		}
		# endif
	}
}
