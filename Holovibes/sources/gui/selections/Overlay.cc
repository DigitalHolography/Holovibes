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

#include "Overlay.hh"

namespace holovibes
{
	namespace gui
	{
		HOverlay::HOverlay() :
			Zone(1, 1),
			kOverlay(KindOfOverlay::Zoom),
			rectBuffer{Rectangle(0, 0), Rectangle(0, 0)},
			verticesIndex(0),
			colorIndex(0),
			elemIndex(0),
			Program(nullptr),
			Colors{ {
				{ 0.0f,	0.5f,	0.0f },		// Zoom
				{ 0.557f, 0.4f, 0.85f },	// Average::Signal
				{ 0.f,	0.64f,	0.67f },	// Average::Noise
				{ 1.f,	0.8f,	0.0f },		// Autofocus
				{ 0.f,	0.62f,	1.f },		// Filter2D
				{ 1.f,	0.87f,	0.87f },	// ?SliceZoom?
				{ 1.f,	0.f,	0.f} } },	// Cross
				Enabled(false)
		{}

		HOverlay::~HOverlay()
		{
			if (elemIndex) glDeleteBuffers(1, &elemIndex);
			if (verticesIndex) glDeleteBuffers(1, &verticesIndex);
			if (!Program) delete Program;
		}

		const Rectangle&		HOverlay::getConstZone() const
		{
			return (Zone);
		}

		Rectangle&		HOverlay::getZone()
		{
			return (Zone);
		}

		const KindOfOverlay		HOverlay::getKind() const
		{
			return (kOverlay);
		}

		const Color		HOverlay::getColor() const
		{
			return (Colors[kOverlay]);
		}

		Rectangle		HOverlay::getTexZone(ushort frameSide) const
		{
			return (Rectangle(
				Zone.topLeft() * frameSide / 512,
				Zone.size() * frameSide / 512
			));
		}
		
		Rectangle		HOverlay::getRectBuffer(KindOfOverlay k) const
		{
			return (rectBuffer[(k == Noise)]);
		}

		const bool		HOverlay::isEnabled() const
		{
			return (Enabled);
		}

		void			HOverlay::setEnabled(bool b)
		{
			Enabled = b;
		}

		void			HOverlay::setKind(KindOfOverlay k)
		{
			kOverlay = k;
			if (kOverlay == Signal || kOverlay == Noise)
				Enabled = true;
			setColor();
		}

		/* ------------------------------- */

		void	HOverlay::initShaderProgram()
		{
			initializeOpenGLFunctions();
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/vertex.overlay.glsl");
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/fragment.color.glsl");
			if (!Program->bind())
				std::cerr << "[Error] " << Program->log().toStdString() << std::endl;
			initBuffers();
			Program->release();
		}

		void	HOverlay::initBuffers()
		{
			rectBuffer.fill(Rectangle(0, 0));
			const float vertices[] = {
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f,
				// ---------
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f
			};
			glGenBuffers(1, &verticesIndex);
			glBindBuffer(GL_ARRAY_BUFFER, verticesIndex);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
			glDisableVertexAttribArray(2);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			const float colorData[] = {
				0.f, 0.5f, 0.f,
				0.f, 0.5f, 0.f,
				0.f, 0.5f, 0.f,
				0.f, 0.5f, 0.f,
				// ---------
				0.f, 0.64f, 0.67f,
				0.f, 0.64f, 0.67f,
				0.f, 0.64f, 0.67f,
				0.f, 0.64f, 0.67f
			};
			glGenBuffers(1, &colorIndex);
			glBindBuffer(GL_ARRAY_BUFFER, colorIndex);
			glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(3);
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
			glDisableVertexAttribArray(3);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			const GLuint elements[] = {
				0, 1, 2,
				2, 3, 0,
				4, 5, 6,
				6, 7, 4
			};
			glGenBuffers(1, &elemIndex);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}

		void	HOverlay::resetVerticesBuffer()
		{
			if (Program)
			{
				Program->bind();
				rectBuffer.fill(Rectangle(0, 0));
				const float vertices[] = {
					0.f, 0.f,
					0.f, 0.f,
					0.f, 0.f,
					0.f, 0.f,
					// ---------
					0.f, 0.f,
					0.f, 0.f,
					0.f, 0.f,
					0.f, 0.f
				};
				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program->release();
			}
		}

		void	HOverlay::setZoneBuffer(int side)
		{
			if (Program)
			{
				Program->bind();
				const float x0 = ((static_cast<float>(Zone.topLeft().x()) - (side * 0.5f)) / side) * 2.f;
				const float y0 = (-((static_cast<float>(Zone.topLeft().y()) - (side * 0.5f)) / side)) * 2.f;
				const float x1 = ((static_cast<float>(Zone.bottomRight().x()) - (side * 0.5f)) / side) * 2.f;
				const float y1 = (-((static_cast<float>(Zone.bottomRight().y()) - (side * 0.5f)) / side)) * 2.f;
				const auto offset = (kOverlay == Noise) ? (8 * sizeof(float)) : 0;

				rectBuffer[(kOverlay == Noise)] = Zone;
				const float subVertices[] = {
					x0, y0,
					x1, y0,
					x1, y1,
					x0, y1
				};
				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex);
				glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(subVertices), subVertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program->release();
			}
		}

		void	HOverlay::setZoneBuffer(int side, Rectangle rect, KindOfOverlay k)
		{
			if (Program)
			{
				Program->bind();
				const float x0 = ((static_cast<float>(rect.topLeft().x()) - (side * 0.5)) / side) * 2.;
				const float y0 = (-((static_cast<float>(rect.topLeft().y()) - (side * 0.5)) / side)) * 2.;
				const float x1 = ((static_cast<float>(rect.bottomRight().x()) - (side * 0.5)) / side) * 2.;
				const float y1 = (-((static_cast<float>(rect.bottomRight().y()) - (side * 0.5)) / side)) * 2.;
				const auto offset = (k == Noise) ? (8 * sizeof(float)) : 0;

				rectBuffer[(k == Noise)] = rect;
				const float subVertices[] = {
					x0, y0,
					x1, y0,
					x1, y1,
					x0, y1
				};

				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex);
				glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(subVertices), subVertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program->release();
			}
		}

		void	HOverlay::initCrossBuffer()
		{
			if (Program)
			{
				Program->bind();
				const float vertices[] = {
					0.f, 1.f,
					0.f, -1.f,
					-1.f, 0.f,
					1.f, 0.f
				};
				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program->release();
			}
		}

		void	HOverlay::setCrossBuffer(QPoint pos, QSize frame)
		{
			if (Program)
			{
				Program->bind();
				const float newX = ((static_cast<float>(pos.x()) - (frame.width() * 0.5)) / frame.width()) * 2.;
				const float newY = (-((static_cast<float>(pos.y()) - (frame.height() * 0.5)) / frame.height())) * 2.;
				const float vertices[] = {
					newX, 1.f,
					newX, -1.f,
					-1.f, newY,
					1.f, newY
				};
				glBindBuffer(GL_ARRAY_BUFFER, verticesIndex);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program->release();
			}
		}

		void	HOverlay::setColor()
		{
			if (Program && kOverlay != Noise)
			{
				Program->bind();
				const Color tab = Colors[kOverlay];
				const float color[] = {
					tab[0], tab[1], tab[2],
					tab[0], tab[1], tab[2],
					tab[0], tab[1], tab[2],
					tab[0], tab[1], tab[2],
				};
				glBindBuffer(GL_ARRAY_BUFFER, colorIndex);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(color), color);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				Program->release();
			}
		}

		void	HOverlay::drawSelections()
		{
			Program->bind();
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex);
			glEnableVertexAttribArray(2);
			glEnableVertexAttribArray(3);

			glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);

			glDisableVertexAttribArray(3);
			glDisableVertexAttribArray(2);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			Program->release();
		}

		void	HOverlay::drawCross()
		{
			Program->bind();
			glEnableVertexAttribArray(2);
			glEnableVertexAttribArray(3);

			glDrawArrays(GL_LINES, 0, 8);

			glDisableVertexAttribArray(3);
			glDisableVertexAttribArray(2);
			Program->release();
		}

		/* ------------------------------- */

		void	HOverlay::press(QPoint pos)
		{
			Zone.setTopLeft(pos);
			Zone.setBottomRight(Zone.topLeft());
			Enabled = true;
		}

		void	HOverlay::move(QPoint pos, int side)
		{
			Zone.setBottomRight(pos);
			if (kOverlay == Filter2D)
			{
				const int max = std::max(Zone.width(), Zone.height());
				Zone.setBottomRight(QPoint(
					Zone.topLeft().x() +
					max * ((Zone.topLeft().x() < Zone.bottomRight().x()) * 2 - 1),
					Zone.topLeft().y() +
					max * ((Zone.topLeft().y() < Zone.bottomRight().y()) * 2 - 1)
				));
			}
			if (Enabled)
				setZoneBuffer(side);
		}

		void	HOverlay::release()
		{
			Zone.checkCorners();
			if (kOverlay != Signal && kOverlay != Noise)
			{
				Enabled = false;
				resetVerticesBuffer();
			}
			else
			{
				setKind((kOverlay == Signal) ? Noise : Signal);
			}
		}
	}
}
