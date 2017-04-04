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
#include "frame_desc.hh"
#include "Rectangle.hh"

namespace holovibes
{
	namespace gui
	{
		using KindOfOverlay =
			enum
		{
			Zoom,
			// Average
			Signal,
			Noise,
			// -------
			Autofocus,
			Filter2D,
			SliceZoom,
			Cross
		};
		using Color = std::array<float, 3>;
		using ColorArray = std::array<Color, 7>;

		class HOverlay : protected QOpenGLFunctions
		{
		public:
			HOverlay();
			virtual ~HOverlay();

			const Rectangle&		getConstZone()	const;
			Rectangle&				getZone();
			Rectangle				getTexZone(ushort frameSide) const;
			Rectangle				getRectBuffer(KindOfOverlay k = Zoom) const;

			const KindOfOverlay		getKind()	const;
			const Color				getColor()	const;
			const bool				isEnabled() const;
			void					setEnabled(bool b);

			void initShaderProgram();
			void initBuffers();

			void setZoneBuffer();
			void setZoneBuffer(Rectangle rect, KindOfOverlay k);
			void resetVerticesBuffer();
			void initCrossBuffer();
			void setCrossBuffer(QPoint pos, QSize frame);
			void drawSelections();
			void drawCross();

			void setKind(KindOfOverlay k);
			void setColor();

			void press(QPoint pos);
			void move(QPoint pos);
			void release();

		protected:
			Rectangle				Zone;
			KindOfOverlay			kOverlay;
			std::array<Rectangle, 2>	rectBuffer;
			GLuint					verticesIndex, colorIndex, elemIndex;
			QOpenGLShaderProgram*	Program;
			ColorArray				Colors;

		private:
			bool	Enabled;
		};
	}
}
