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
		class DirectWindow : public BasicOpenGLWindow
		{
		public:
			DirectWindow(QPoint p, QSize s, Queue& q);
			DirectWindow(QPoint p, QSize s, Queue& q, KindOfView k);
			virtual ~DirectWindow();

			units::RectFd	getSignalZone() const;
			units::RectFd	getNoiseZone() const;
			void		setSignalZone(units::RectFd signal);
			void		setNoiseZone(units::RectFd noise);

			void	zoomInRect(units::RectOpengl zone);

		protected:
			int	texDepth, texType;

			virtual void	initShaders();
			virtual void	initializeGL();
			virtual void	resizeGL(int width, int height);
			virtual void	paintGL();
			
			void	mousePressEvent(QMouseEvent* e);
			void	mouseMoveEvent(QMouseEvent* e);
			void	mouseReleaseEvent(QMouseEvent* e);
		};
	}
}
