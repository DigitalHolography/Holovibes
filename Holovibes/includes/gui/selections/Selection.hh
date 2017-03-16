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

namespace gui
{
	using KindOfSelection =
	enum
	{
		None = -1,
		Zoom,
		Average,	// Signal == 1, Noise == 2
		Autofocus = 3,
		Filter2D,
		SliceZoom,
	};
	using Color = std::array<float, 4>;
	using ColorArray = std::array<Color, 6>;

	class Selection : protected QOpenGLFunctions
	{
		public:
			Selection();
			virtual ~Selection();

			const Rectangle&		getZone()	const;
			const KindOfSelection	getKind()	const;
			const Color				getColor()	const;
			const bool				isEnabled() const;

			void initShaderProgram();
			void initBuffers();
			void setZoneBuffer();
			void setUniformColor();
			void draw();

			void setKind(KindOfSelection k);

			void press(QPoint pos);
			void move(QPoint pos);
			void release();

		protected:
			Rectangle				Zone;
			KindOfSelection			kSelection;
			GLuint					zoneBuffer, elemBuffer;
			QOpenGLShaderProgram	*Program;
			ColorArray				Colors;

		private:
			bool	Enabled;
			void	resetZoneBuffer();
	};
}
