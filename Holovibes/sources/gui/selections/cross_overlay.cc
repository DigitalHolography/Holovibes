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

#include "cross_overlay.hh"

namespace holovibes
{
	namespace gui
	{
		CrossOverlay::CrossOverlay(WindowKind view)
			: Overlay(KindOfOverlay::Cross)
			, doubleCross_(false)
		{
			color_ = { 1.f, 0.f, 0.f };
		}

		void CrossOverlay::setBuffer(QPoint pos, QSize frame)
		{
			if (Program_)
			{
				Program_->bind();
				const float newX = ((static_cast<float>(pos.x()) - (frame.width() * 0.5f)) / frame.width()) * 2.f;
				const float newY = (-((static_cast<float>(pos.y()) - (frame.height() * 0.5f)) / frame.height())) * 2.f;
				const float vertices[] = {
					newX, 1.f,
					newX, -1.f,
					-1.f, newY,
					1.f, newY,
					newX, 1.f,
					newX, -1.f,
					-1.f, newY,
					1.f, newY,
				};
				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program_->release();
			}
		}

		void CrossOverlay::setDoubleBuffer(QPoint pos, QPoint pos2, QSize frame)
		{
			if (Program_)
			{
				Program_->bind();
				const float newX = ((static_cast<float>(pos.x()) - (frame.width() * 0.5f)) / frame.width()) * 2.f;
				const float newY = (-((static_cast<float>(pos.y()) - (frame.height() * 0.5f)) / frame.height())) * 2.f;
				const float newX2 = ((static_cast<float>(pos2.x()) - (frame.width() * 0.5f)) / frame.width()) * 2.f;
				const float newY2 = (-((static_cast<float>(pos2.y()) - (frame.height() * 0.5f)) / frame.height())) * 2.f;
				const float vertices[] = {
					newX, 1.f,
					newX, -1.f,
					newX2, 1.f,
					newX2, -1.f,
					-1.f, newY,
					1.f, newY,
					-1.f, newY2,
					1.f, newY2,
				};
				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program_->release();
			}
			doubleCross_ = true;
		}

		void CrossOverlay::init()
		{
			if (Program_)
			{
				Program_->bind();
				const float vertices[] = {
					0.f, 1.f,
					0.f, -1.f,
					-1.f, 0.f,
					1.f, 0.f
				};
				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program_->release();
			}
		}

		void CrossOverlay::draw()
		{
		}
	}
}
