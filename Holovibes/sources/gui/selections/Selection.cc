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

#include "Selection.hh"

namespace gui
{
	Selection::Selection() :
		Zone(1, 1),
		kSelection(KindOfSelection::Zoom),
		zoneBuffer(0),
		elemBuffer(0),
		Program(nullptr),
		Colors{ {
			{ 0.0f,	0.5f, 0.0f },			// Zoom
			// Average::Signal
			//{ 1.0f, 0.0f, 0.5f }, // Fushia
			{ 0.557f, 0.4f, 0.85f }, // Mauve
			// Average::Noise
			//{ 0.26f, 0.56f, 0.64f }, // Gris-Turquoise
			{ 0.f, 0.64f, 0.67f }, // Turquoise
			{ 1.0f,	0.8f, 0.0f },			// Autofocus
			{ 0.f,	0.62f, 1.f },			// Filter2D
			{ 1.0f,	0.87f, 0.87f } } },	// SliceZoom
		Enabled(false)
	{}

	Selection::~Selection()
	{
		if (elemBuffer) glDeleteBuffers(1, &elemBuffer);
		if (zoneBuffer) glDeleteBuffers(1, &zoneBuffer);
		if (!Program) delete Program;
	}

	const Rectangle&		Selection::getConstZone() const
	{
		return (Zone);
	}

	Rectangle&				Selection::getZone()
	{
		return (Zone);
	}

	const KindOfSelection	Selection::getKind() const
	{
		return (kSelection);
	}

	const Color				Selection::getColor() const
	{
		return (Colors[kSelection]);
	}

	Rectangle				Selection::getTexZone(ushort frameSide) const
	{
		return (Rectangle(
			Zone.topLeft() * frameSide / 512,
			Zone.size() * frameSide / 512
		));
	}

	const bool				Selection::isEnabled() const
	{
		return (Enabled);
	}

	void					Selection::setEnabled(bool b)
	{
		Enabled = b;
	}

	void					Selection::setKind(KindOfSelection k)
	{
		kSelection = k;
		setZoneColor();
	}

	/* ------------------------------- */

	void	Selection::initShaderProgram()
	{
		initializeOpenGLFunctions();
		Program = new QOpenGLShaderProgram();
		Program->addShaderFromSourceFile(QOpenGLShader::Vertex,
			"shaders/selections.vertex.glsl");
		Program->addShaderFromSourceFile(QOpenGLShader::Fragment,
			"shaders/selections.fragment.glsl");
		if (!Program->bind())
			std::cerr << "[Error] " << Program->log().toStdString() << std::endl;
		initBuffers();
		Program->release();
	}
	
	void	Selection::initBuffers()
	{
		const float data[] = {
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
		glGenBuffers(1, &zoneBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, zoneBuffer);
		glBufferData(GL_ARRAY_BUFFER, nbVertices * sizeof(float), data, GL_DYNAMIC_DRAW);
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
		glGenBuffers(1, &colorBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
		glBufferData(GL_ARRAY_BUFFER, 24 * sizeof(float), colorData, GL_DYNAMIC_DRAW);
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
		glGenBuffers(1, &elemBuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, nbElements * sizeof(GLuint), elements, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		
		/*const float nColor[] = {
			Colors[kSelection][0],
			Colors[kSelection][1],
			Colors[kSelection][2]
		};
		glUniform3fv(glGetUniformLocation(Program->programId(), "color"), 1, nColor);*/
	}

	void	Selection::resetZoneBuffer()
	{
		if (Program)
		{
			Program->bind();
			const float data[] = {
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
			glBindBuffer(GL_ARRAY_BUFFER, zoneBuffer);
			glBufferSubData(GL_ARRAY_BUFFER, 0, nbVertices * sizeof(float), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			Program->release();
		}
	}

	void	Selection::setZoneBuffer()
	{
		if (Program)
		{
			Program->bind();
			const float x0 = ((static_cast<float>(Zone.topLeft().x()) - (512 * 0.5)) / 512) * 2.;
			const float y0 = (-((static_cast<float>(Zone.topLeft().y()) - (512 * 0.5)) / 512)) * 2.;
			const float x1 = ((static_cast<float>(Zone.bottomRight().x()) - (512 * 0.5)) / 512) * 2.;
			const float y1 = (-((static_cast<float>(Zone.bottomRight().y()) - (512 * 0.5)) / 512)) * 2.;
			const float data[] = {
				x0, y0,
				x1, y0,
				x1, y1,
				x0, y1
			};
			glBindBuffer(GL_ARRAY_BUFFER, zoneBuffer);
			glBufferSubData(GL_ARRAY_BUFFER, 
				(kSelection == Noise) ? (8 * sizeof(float)) : 0,
				8 * sizeof(float), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			Program->release();
		}
	}

	void	Selection::setZoneColor()
	{
		if (Program && kSelection != Noise)
		{
			Program->bind();
			Color tab = Colors[kSelection];
			const float data[] = {
				tab[0], tab[1], tab[2],
				tab[0], tab[1], tab[2],
				tab[0], tab[1], tab[2],
				tab[0], tab[1], tab[2]
			};
			glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
			glBufferSubData(GL_ARRAY_BUFFER, 0, 12 * sizeof(float), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			Program->release();
		}
	}

	void	Selection::draw()
	{
		Program->bind();
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemBuffer);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(3);

		glDrawElements(GL_TRIANGLES, nbElements, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(2);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		Program->release();
	}

	/* ------------------------------- */

	void	Selection::press(QPoint pos)
	{
		Zone.setTopLeft(pos);
		Zone.setBottomRight(Zone.topLeft());
		Enabled = true;
		
	}

	void	Selection::move(QPoint pos)
	{
		Zone.setBottomRight(pos);
		if (kSelection == Filter2D)
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
			setZoneBuffer();
	}

	void	Selection::release()
	{
		Zone.checkCorners();
		if (kSelection != Signal && kSelection != Noise)
		{
			Enabled = false;
			resetZoneBuffer();
		}
		else
			setKind((kSelection == Signal) ? Noise : Signal);
	}
}
