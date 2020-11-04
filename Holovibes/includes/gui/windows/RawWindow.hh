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
* Qt window used to display the input frames. */
#pragma once

#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		class SliceWindow;
		class RawWindow : public BasicOpenGLWindow
		{
		public:
			RawWindow(QPoint p, QSize s, std::unique_ptr<Queue>& q, KindOfView k = Raw);
			virtual ~RawWindow();

			void zoomInRect(units::RectOpengl zone);
			void setRatio(float ratio_);

			bool is_resize_call() const;
			void set_is_resize(bool b);

		protected:
			int	texDepth, texType;

			int old_width = -1;
			int old_height = -1;
			// it represents width/height of the Raw window
			float ratio = 0.0f;

			// bool represent if we are resizing the window or creating one
			bool is_resize = true;

			const float translation_step_ = 0.05f;

			virtual void	initShaders() override;
			virtual void	initializeGL() override;
			virtual void	resizeGL(int width, int height) override;
			virtual void	paintGL() override;

			void mousePressEvent(QMouseEvent* e);
			void mouseMoveEvent(QMouseEvent* e);
			void mouseReleaseEvent(QMouseEvent* e);
			void keyPressEvent(QKeyEvent *e) override;
			void wheelEvent(QWheelEvent *e) override;
		};
	}
}
