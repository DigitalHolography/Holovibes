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
		OverlayManager::OverlayManager(WindowKind view)
			: view_(view)
			, current_overlay_(nullptr)
		{

		}

		OverlayManager::~OverlayManager()
		{}

		void OverlayManager::create_overlay(std::shared_ptr<Overlay> new_overlay)
		{
			overlays_.push_back(new_overlay);
			current_overlay_ = new_overlay;
		}

		void OverlayManager::create_autofocus()
		{
			create_overlay(std::make_shared<AutofocusOverlay>());
		}

		void OverlayManager::create_zoom()
		{
			create_overlay(std::make_shared<ZoomOverlay>());
		}

		void OverlayManager::create_filter2D()
		{
			create_overlay(std::make_shared<Filter2DOverlay>());
		}

		void OverlayManager::create_noise()
		{
			create_overlay(std::make_shared<NoiseOverlay>());
		}

		void OverlayManager::create_signal()
		{
			create_overlay(std::make_shared<SignalOverlay>());
		}

		void OverlayManager::create_cross()
		{
			create_overlay(std::make_shared<CrossOverlay>(view_));
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
				current_overlay_->release(frame);
		}

		void OverlayManager::disable_all(KindOfOverlay ko)
		{
			for (auto o : overlays_)
				if (o->getKind() == ko)
					o->disable();
		}
	}
}
