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

#include "DirectWindow.hh"
#include "texture_update.cuh"

namespace gui
{
	DirectWindow::DirectWindow(QPoint p, QSize s, holovibes::Queue& q) :
		BasicOpenGLWindow(p, s, q, KindOfView::Direct)
	{}

	DirectWindow::DirectWindow(QPoint p, QSize s, holovibes::Queue& q, KindOfView k) :
		BasicOpenGLWindow(p, s, q, k)
	{}

	DirectWindow::~DirectWindow()
	{}

	void DirectWindow::initShaders()
	{
		Program = new QOpenGLShaderProgram();
		Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/direct.vertex.glsl");
		Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/direct.fragment.glsl");
		if (!Program->bind()) std::cerr << "[Error] " << Program->log().toStdString() << '\n';
	}

	void DirectWindow::initializeGL()
	{
		makeCurrent();
		initializeOpenGLFunctions();
		glClearColor(0.128f, 0.128f, 0.128f, 1.0f);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);

		#pragma region Shaders & Vao
		initShaders();
		if (!Vao.create()) std::cerr << "[Error] Vao create() fail\n";
		Vao.bind();
		#pragma endregion

		#pragma region Texture
		unsigned int size = Fd.frame_size();		
		glGenBuffers(1, &Pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
		glPixelStorei(GL_UNPACK_SWAP_BYTES,
			(Fd.endianness == camera::BIG_ENDIAN) ?
			GL_TRUE : GL_FALSE);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cuResource, Pbo,
			cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
		/* -------------------------------------------------- */
		glGenTextures(1, &Tex);
		glBindTexture(GL_TEXTURE_2D, Tex);
		texDepth = (Fd.depth == 1) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT;
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, Fd.width, Fd.height, 0, GL_RED, texDepth, nullptr);
		
		glUniform1i(glGetUniformLocation(Program->programId(), "tex"), 0);

		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
		glBindTexture(GL_TEXTURE_2D, 0);
		#pragma endregion

		#pragma region Vertex Buffer Object
		const float	data[16] = {
			// Top-left
			-1.f, 1.f,		// vertex coord (-1.0f <-> 1.0f)
			0.0f, 0.0f,		// texture coord (0.0f <-> 1.0f)
			// Top-right
			1.f, 1.f,
			1.f, 0.0f,
			// Bottom-right
			1.f, -1.f,
			1.f, 1.f,
			// Bottom-left
			-1.f, -1.f,
			0.0f, 1.f
		};
		glGenBuffers(1, &Vbo);
		glBindBuffer(GL_ARRAY_BUFFER, Vbo);
		glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(float), data, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
			reinterpret_cast<void*>(2 * sizeof(float)));

		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		#pragma endregion

		#pragma region Element Buffer Object
		const GLuint elements[6] = {
			0, 1, 2,
			2, 3, 0
		};
		glGenBuffers(1, &Ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLuint), elements, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		#pragma endregion

		glUniform1f(glGetUniformLocation(Program->programId(), "scale"), Scale);
		glUniform2f(glGetUniformLocation(Program->programId(), "translate"), Translate[0], Translate[1]);
		if (kView == Hologram)
		{
			glUniform1i(glGetUniformLocation(Program->programId(), "flip"), 0);
			glUniform1f(glGetUniformLocation(Program->programId(), "angle"), 0 * (M_PI / 180.f));
		}
		Program->release();
		zoneSelected.initShaderProgram();
		Vao.release();
		glViewport(0, 0, width(), height());
		startTimer(DisplayRate);
	}

	void DirectWindow::resizeGL(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void DirectWindow::paintGL()
	{
		makeCurrent();
		glClear(GL_COLOR_BUFFER_BIT);

		Vao.bind();
		Program->bind();

		cudaGraphicsMapResources(1, &cuResource, cuStream);
		cudaGraphicsResourceGetMappedPointer(&cuPtrToPbo, &sizeBuffer, cuResource);
		cudaMemcpy(cuPtrToPbo, Queue.get_last_images(1), sizeBuffer, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &cuResource, cuStream);
		cudaStreamSynchronize(cuStream);

		glBindTexture(GL_TEXTURE_2D, Tex);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Pbo);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Fd.width, Fd.height, GL_RED, texDepth, nullptr);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
		
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);		

		Program->release();
		if (zoneSelected.isEnabled())
			zoneSelected.draw();
		Vao.release();
	}

	void DirectWindow::mousePressEvent(QMouseEvent* e)
	{
		if (slicesAreLocked.load())
		{
			if (e->button() == Qt::LeftButton)
				zoneSelected.press(e->pos());
		}
	}

	void DirectWindow::mouseMoveEvent(QMouseEvent* e)
	{
		if (e->buttons() == Qt::LeftButton)
			zoneSelected.move(e->pos());
	}

	void DirectWindow::mouseReleaseEvent(QMouseEvent* e)
	{
		if (e->button() == Qt::LeftButton)
		{
			//if (kView == Direct)
				zoneSelected.release();
			if (zoneSelected.getConstZone().topLeft() !=
				zoneSelected.getConstZone().bottomRight())
			{
				if (zoneSelected.getKind() == Zoom)
					zoomInRect(zoneSelected.getConstZone());
			}
		}
		else if (e->button() == Qt::RightButton)
			resetTransform();
	}

	void	DirectWindow::zoomInRect(Rectangle zone)
	{
		const QPoint center = zone.center();

		Translate[0] += ((static_cast<float>(center.x()) / static_cast<float>(width())) - 0.5) / Scale;
		Translate[1] += ((static_cast<float>(center.y()) / static_cast<float>(height())) - 0.5) / Scale;
		setTranslate();

		const float xRatio = static_cast<float>(width()) / static_cast<float>(zone.width());
		const float yRatio = static_cast<float>(height()) / static_cast<float>(zone.height());

		Scale = (xRatio < yRatio ? xRatio : yRatio) * Scale;
		setScale();
	}

	void	DirectWindow::wheelEvent(QWheelEvent *e)
	{
		if (e->x() < width() && e->y() < height())
		{
			const float xGL = (static_cast<float>(e->x() - width() / 2)) / static_cast<float>(width()) * 2.f;
			const float yGL = -((static_cast<float>(e->y() - height() / 2)) / static_cast<float>(height())) * 2.f;
			if (e->angleDelta().y() > 0)
			{
				Scale += 0.1f * Scale;
				setScale();
				Translate[0] += xGL * 0.1 / Scale;
				Translate[1] += -yGL * 0.1 / Scale;
				setTranslate();
			}
			else if (e->angleDelta().y() < 0)
			{
				Scale -= 0.1f * Scale;
				if (Scale < 1.f)
					resetTransform();
				else
				{
					setScale();
					Translate[0] -= -xGL * 0.1 / Scale;
					Translate[1] -= yGL * 0.1 / Scale;
					setTranslate();
				}
			}
		}
	}

}