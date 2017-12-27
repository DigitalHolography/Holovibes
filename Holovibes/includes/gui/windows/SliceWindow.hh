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
* Qt window containing the XZ or YZ view of the hologram. */
#pragma once

#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		class MainWindow;

		class SliceWindow : public BasicOpenGLWindow
		{
		public:
			SliceWindow(QPoint p, QSize s, std::unique_ptr<Queue>& q, KindOfView k, MainWindow *main_window = nullptr);
			virtual ~SliceWindow();
			void make_pixel_square();
			void setTransform() override;

			void adapt();
			
		protected:
			cudaArray_t				cuArray;
			cudaResourceDesc		cuArrRD;
			cudaSurfaceObject_t		cuSurface;
			MainWindow *main_window_;
			
			virtual void	initShaders() override;
			virtual void	initializeGL() override;
			virtual void	paintGL() override;

			void changeTexture();

			void mousePressEvent(QMouseEvent*) override;
			void mouseMoveEvent(QMouseEvent*) override;
			void mouseReleaseEvent(QMouseEvent*) override;
			void focusInEvent(QFocusEvent*) override;

			bool changeTexture_ = false;
		};
	}
}
