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

#include "icompute.hh"
#include "compute_descriptor.hh"
#include "info_manager.hh"
#include "DirectWindow.hh"

namespace holovibes
{
	namespace gui
	{
		using SharedPipe = std::shared_ptr<ICompute>;
		using CDescriptor = ComputeDescriptor;

		class HoloWindow : public DirectWindow
		{
		public:
			HoloWindow(QPoint p, QSize s, Queue& q,
				SharedPipe ic, CDescriptor& cd);
			virtual ~HoloWindow();

		protected:
			SharedPipe		Ic;
			CDescriptor&	Cd;

			virtual void	initShaders();

			void	mousePressEvent(QMouseEvent* e);
			void	mouseMoveEvent(QMouseEvent* e);
			void	mouseReleaseEvent(QMouseEvent* e);

			void	updateCursorPosition(QPoint pos);
		};
	}
}
