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

#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		class MainWindow;

		class SliceWindow : public BasicOpenGLWindow
		{
		public:
			SliceWindow(QPoint p, QSize s, Queue& q, KindOfView k, MainWindow *main_window = nullptr);
			virtual ~SliceWindow();
			void	setPIndex(ushort pId);
			
		protected:
			cudaArray_t				cuArray;
			cudaResourceDesc		cuArrRD;
			cudaSurfaceObject_t		cuSurface;
			ushort		pIndex;
			MainWindow *main_window_;
			QPoint last_clicked;
			QPoint mouse_position;
			bool is_pslice_locked = true;
			
			virtual void	initShaders();
			virtual void	initializeGL();
			virtual void	paintGL();

			void mousePressEvent(QMouseEvent*);
			void mouseMoveEvent(QMouseEvent*);
			void mouseReleaseEvent(QMouseEvent*);
			void focusInEvent(QFocusEvent*);
			void keyPressEvent(QKeyEvent* e);
		};
	}
}
