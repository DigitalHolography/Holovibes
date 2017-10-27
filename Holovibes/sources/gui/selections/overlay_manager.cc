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
#include "stabilization_overlay.hh"
#include "slice_cross_overlay.hh"
#include "filter2d_overlay.hh"
#include "strip_overlay.hh"

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

		template <>
		void OverlayManager::create_overlay<Autofocus>()
		{
			if (!set_current(Autofocus))
				create_overlay(std::make_shared<AutofocusOverlay>(parent_));
		}

		template <>
		void OverlayManager::create_overlay<Noise>()
		{
			if (!set_current(Noise))
				create_overlay(std::make_shared<NoiseOverlay>(parent_));
		}

		template <>
		void OverlayManager::create_overlay<Signal>()
		{
			if (!set_current(Signal))
				create_overlay(std::make_shared<SignalOverlay>(parent_));
		}

		template <>
		void OverlayManager::create_overlay<Cross>()
		{
			if (!set_current(Cross))
				create_overlay(std::make_shared<CrossOverlay>(parent_));
		}

		template <>
		void OverlayManager::create_overlay<Stabilization>()
		{
			if (!set_current(Stabilization))
				create_overlay(std::make_shared<StabilizationOverlay>(parent_));
		}

		template <>
		void OverlayManager::create_overlay<SliceCross>()
		{
			if (!set_current(SliceCross))
				create_overlay(std::make_shared<SliceCrossOverlay>(parent_));
		}

		void OverlayManager::create_strip_overlay(Component& component,
			std::atomic<ushort>& nsamples,
			Color color)
		{
			auto strip = std::make_shared<StripOverlay>(parent_, component, nsamples, color);
			strip->initProgram();
			overlays_.push_back(strip);
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

		void OverlayManager::set_zone(ushort frameside, Rectangle zone, KindOfOverlay ko)
		{
			if (ko == Noise)
				create_overlay<Noise>();
			else if (ko == Signal)
				create_overlay<Signal>();
			else
				create_default();
			current_overlay_->setZone(zone, frameside);
			create_default();
		}

		void OverlayManager::press(QPoint pos)
		{
			if (current_overlay_)
				current_overlay_->press(pos);
		}

		void OverlayManager::keyPress(QPoint pos)
		{
			if (current_overlay_)
				current_overlay_->keyPress(pos);
		}

		void OverlayManager::move(QPoint pos)
		{
			if (current_overlay_)
				current_overlay_->move(pos);
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
				else if (current_overlay_->getKind() == Stabilization)
					create_default();
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
			for (auto o : overlays_) {
				if (o->isActive() && o->isDisplayed())
					o->draw();
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
			case Direct:
			case Hologram:
				if (!set_current(Cross))
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
			auto cross = dynamic_cast<CrossOverlay *>(current_overlay_.get());
			if (!cross)
				return false;
			cross->setBuffer(pos, frame);
			return true;
		}

		bool OverlayManager::setDoubleCrossBuffer(QPoint pos, QPoint pos2, QSize frame)
		{
			auto cross = dynamic_cast<CrossOverlay *>(current_overlay_.get());
			if (!cross)
				return false;
			cross->setDoubleBuffer(pos, pos2, frame);
			return true;
		}

		# ifdef _DEBUG
		void OverlayManager::printVector()
		{
			std::cout << std::endl;
			std::cout << "Current overlay :" << std::endl;
			current_overlay_->print();
			std::cout << std::endl;
			for (auto o : overlays_)
				o->print();
			std::cout << std::endl;
		}
		# endif
	}
}
