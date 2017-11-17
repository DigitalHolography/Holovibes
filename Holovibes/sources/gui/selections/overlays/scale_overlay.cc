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

#include "scale_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		ScaleOverlay::ScaleOverlay(BasicOpenGLWindow* parent)
			: RectOverlay(KindOfOverlay::Scale, parent)
			, line_alpha_(1.f)
			, elemLineIndex_(0)
		{
			color_ = { 1.f, 1.f, 1.f };
			alpha_ = 1.f;
			display_ = true;
		}

		ScaleOverlay::~ScaleOverlay()
		{
			parent_->makeCurrent();
			glDeleteBuffers(1, &elemLineIndex_);
		}

		void ScaleOverlay::init()
		{
			RectOverlay::init();

			// Set line vertices order
			std::vector<GLuint> elements{
				0, 1,
				1, 2,
				2, 3,
				3, 0
			};

			glGenBuffers(1, &elemLineIndex_);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements.data(), GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}

		void ScaleOverlay::draw()
		{
			parent_->makeCurrent();
			compute_zone();
			setBuffer();

			Vao_.bind();
			Program_->bind();

			glEnableVertexAttribArray(colorShader_);
			glEnableVertexAttribArray(verticesShader_);

			// Drawing two lines
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
			Program_->setUniformValue(Program_->uniformLocation("alpha"), line_alpha_);
			glDrawElements(GL_LINES, 8, GL_UNSIGNED_INT, nullptr);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			// Drawing area between two lines
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
			Program_->setUniformValue(Program_->uniformLocation("alpha"), alpha_);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			glDisableVertexAttribArray(verticesShader_);
			glDisableVertexAttribArray(colorShader_);

			Program_->release();
			Vao_.release();
		}

		void ScaleOverlay::setBuffer()
		{
			// Compute zone resizing

			// Updating opengl buffer
			RectOverlay::setBuffer();
		}
	}
}
