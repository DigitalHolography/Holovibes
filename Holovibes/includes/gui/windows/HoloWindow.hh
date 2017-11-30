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

/*! \file
*
* Qt window displaying the hologram in XY view. */
#pragma once

#include "icompute.hh"
#include "DirectWindow.hh"


namespace holovibes
{
	namespace gui
	{
		class MainWindow;
		using SharedPipe = std::shared_ptr<ICompute>;

		class HoloWindow : public DirectWindow
		{
		public:
			HoloWindow(QPoint p, QSize s, Queue& q, SharedPipe ic, std::unique_ptr<SliceWindow>& xy, std::unique_ptr<SliceWindow>& yy, MainWindow *main_window = nullptr);
			virtual ~HoloWindow();

			void update_slice_transforms();

			SharedPipe getPipe();

			void	update_stft_zoom_buffer(units::RectFd zone_);
			void	resetTransform() override;

		protected:
			SharedPipe		Ic;

			virtual void	initShaders() override;

			void	focusInEvent(QFocusEvent *e) override;
		private:
			MainWindow *main_window_;

			std::unique_ptr<SliceWindow>& xz_slice_;
			std::unique_ptr<SliceWindow>& yz_slice_;
		};
	}
}
