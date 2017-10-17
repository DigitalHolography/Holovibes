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

#include <array>
#include <iostream>
#include <QOpenGLShaderProgram.h>
#include <QOpenGLFunctions.h>
#include <memory>
#include "frame_desc.hh"
#include "Rectangle.hh"
#include "BasicOpenGLWindow.hh"
#include "DirectWindow.hh"
#include "HoloWindow.hh"

namespace holovibes
{
	namespace gui
	{

		using Color = std::array<float, 3>;

		class Overlay : protected QOpenGLFunctions
		{
		public:
			Overlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);
			virtual ~Overlay();

			const Rectangle&		getZone()	const;

			const KindOfOverlay		getKind()		const;
			const Color				getColor()		const;
			const bool				isDisplayed()	const;
			const bool				isActive()		const;
			void					disable();

			void initProgram();

			//void setZone(int side, Rectangle rect);

			virtual void init() = 0;
			virtual void draw() = 0;

			void press(QPoint pos);
			virtual void move(QPoint pos, QSize size) = 0;
			virtual void release(ushort frameside) = 0;

		protected:
			Rectangle				zone_;
			KindOfOverlay			kOverlay_;
			GLuint					verticesIndex_, colorIndex_, elemIndex_;
			QOpenGLShaderProgram*	Program_;
			Color					color_;
			bool					active_;
			bool					display_;
			std::shared_ptr<BasicOpenGLWindow> parent_;
		};
	}
}
