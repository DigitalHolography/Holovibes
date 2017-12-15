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
* Overlay displaying a strip of color in function of the parent window. */
#pragma once

#include "rect_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		class RainbowOverlay : public RectOverlay
		{
		public:
			RainbowOverlay(BasicOpenGLWindow* parent, double red, double blue, std::atomic<ushort>& nsamples, float alpha = 0.3f)
				: RectOverlay(KindOfOverlay::Strip, parent)
				, nsamples_(nsamples)
				, red_(red), blue_(blue)
			{
				alpha_ = 0.5;// alpha;
				display_ = true;
			}
			
			void compute_zone()
			{
				ushort pmin = red_;
				ushort pmax = blue_;
				units::ConversionData convert(parent_);
				if (parent_->getKindOfView() == SliceXZ)
				{
					units::PointFd topleft(convert, 0, pmin);
					units::PointFd bottomRight(convert, parent_->getFd().width, pmax);
					units::RectFd rect(topleft, bottomRight);
					zone_ = rect;
				}
				else
				{
					units::PointFd topleft(convert, pmin, 0);
					units::PointFd bottomRight(convert, pmax, parent_->getFd().height);
					zone_ = units::RectFd(topleft, bottomRight);
				}
			}
			void release(ushort frameSide) override
			{}

			void move(QMouseEvent *e) override
			{}

			/*void init() override {
				RectOverlay::init();
				// Set color
				const float colorData[] = {
					// vertical area
					1, 0, 0,
					0, 1, 0,
					0, 0, 1,
					0, 1, 1,
					// horizontal area
					color_[0], color_[1], color_[2],
					color_[0], color_[1], color_[2],
					color_[0], color_[1], color_[2],
					color_[0], color_[1], color_[2]
				};
				glBindBuffer(GL_ARRAY_BUFFER, colorIndex_);
				glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_DYNAMIC_DRAW);
				glEnableVertexAttribArray(colorShader_);
				glVertexAttribPointer(colorShader_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
				glDisableVertexAttribArray(colorShader_);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}*/
			/*void draw() override {
				compute_zone();
				setBuffer();
				RectOverlay::draw();
			}*/
			void draw() override {
				compute_zone();
				setBuffer();

				parent_->makeCurrent();
				setBuffer();
				Vao_.bind();
				Program_->bind();
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
				glEnableVertexAttribArray(colorShader_);
				glEnableVertexAttribArray(verticesShader_);
				Program_->setUniformValue(Program_->uniformLocation("alpha"), alpha_);

				glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

				glDisableVertexAttribArray(verticesShader_);
				glDisableVertexAttribArray(colorShader_);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
				Program_->release();
				Vao_.release();
			}
			/*void setBuffer() override {
				parent_->makeCurrent();
				Program_->bind();

				// Normalizing the zone to (-1; 1)
				units::RectOpengl zone_gl = zone_;

				//units::PointOpengl mid_top(parent_->converter_, (zone_gl.x() + zone_gl.right()) / 2, (zone_gl.y() + zone_gl.bottom()) / 2);

				const float subVertices[] = {
					zone_gl.x(), zone_gl.y(),
					//(zone_gl.x() + zone_gl.right()) / 2, (zone_gl.y() + zone_gl.bottom()) / 2
					zone_gl.right(), zone_gl.y(),
					zone_gl.right(), zone_gl.bottom(),
					zone_gl.x(), zone_gl.bottom()
				};

				// Updating the buffer at verticesIndex_ with new coordinates
				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subVertices), subVertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				Program_->release();
			}*/
		private:
			std::atomic<ushort>& nsamples_;
			double red_, blue_;
		};
	}
}