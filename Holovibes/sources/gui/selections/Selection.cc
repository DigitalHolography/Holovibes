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
			{ 0.0f,	0.5f, 0.0f, 0.4f },			// Zoom
			{ 1.0f, 0.0f, 0.5f, 0.4f },			// Average::Signal
			{ 0.26f, 0.56f, 0.64f, 0.4f },		// Average::Noise
			{ 1.0f,	0.8f, 0.0f, 0.4f },			// Autofocus
			{ 0.f,	0.62f, 1.f, 0.4f },			// Filter2D
			{ 1.0f,	0.87f, 0.87f, 0.4f } } },	// SliceZoom
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
		setUniformColor();
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

	void	Selection::resetZoneBuffer()
	{
		if (Program)
		{
			Program->bind();
			const float data[] = {
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f,
				0.f, 0.f
			};
			glBindBuffer(GL_ARRAY_BUFFER, zoneBuffer);
			glBufferSubData(GL_ARRAY_BUFFER, 0, 8 * sizeof(float), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			Program->release();
		}
	}

	void	Selection::initBuffers()
	{
		const float data[] = {
			0.f, 0.f,
			0.f, 0.f,
			0.f, 0.f,
			0.f, 0.f
		};
		glGenBuffers(1, &zoneBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, zoneBuffer);
		glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), data, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
		glDisableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		const GLuint elements[] = {
			0, 1, 2,
			2, 3, 0
		};
		glGenBuffers(1, &elemBuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLuint), elements, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		const float nColor[] = {
			Colors[kSelection][0],
			Colors[kSelection][1],
			Colors[kSelection][2],
			Colors[kSelection][3]
		};
		glUniform4fv(glGetUniformLocation(Program->programId(), "color"), 1, nColor);
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
			glBufferSubData(GL_ARRAY_BUFFER, 0, 8 * sizeof(float), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			Program->release();
		}
	}

	void	Selection::setUniformColor()
	{
		if (Program)
		{
			Program->bind();
			const float nColor[] = {
				Colors[kSelection][0],
				Colors[kSelection][1],
				Colors[kSelection][2],
				Colors[kSelection][3]
			};
			glUniform4fv(glGetUniformLocation(Program->programId(), "color"), 1, nColor);
			Program->release();
		}
	}

	void	Selection::draw()
	{
		Program->bind();
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemBuffer);
		glEnableVertexAttribArray(2);

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

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
		Enabled = false;
		resetZoneBuffer();
	}
}
