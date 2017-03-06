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

#include "texture_update.cuh"
#include "HoloWindow.hh"

/*
	x = ((x - (width * 0.5)) / width) * 2.;
	y = (-((y - (height * 0.5)) / height)) * 2.;
*/

namespace gui
{

	bool BasicOpenGLWindow::sliceLock = false;

	HoloWindow::HoloWindow(QPoint p, QSize s, holovibes::Queue& q, KindOfView k) :
		BasicOpenGLWindow(p, s, q, k),
		Translate{ 0.f, 0.f },
		Scale(1.f),
		kSelection(KindOfSelection::None),
		selectionRect(1, 1),
		selectionColors{ {
			{ 0.0f,	0.5f, 0.0f, 0.4f },			// Zoom
			{ 1.0f, 0.0f, 0.5f, 0.4f },			// Average::Signal
			{ 0.26f, 0.56f, 0.64f, 0.4f },		// Average::Noise
			{ 1.0f,	0.8f, 0.0f, 0.4f },			// Autofocus
			{ 0.9f,	0.7f, 0.1f, 0.4f },			// Filter2D
			{ 1.0f,	0.87f, 0.87f, 0.4f } } }		// SliceZoom
	{
		//auto tab = selectionColors[kSelection];
		sliceLock = false;

	}

	HoloWindow::~HoloWindow()
	{}

	void HoloWindow::setKindOfSelection(KindOfSelection k)
	{
		kSelection = k;
	}

	KindOfSelection HoloWindow::getKindOfSelection() const
	{
		return kSelection;
	}

	void HoloWindow::initializeGL()
	{
		makeCurrent();
		initializeOpenGLFunctions();
		glClearColor(0.128f, 0.128f, 0.128f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		#pragma region Shaders
		Program = new QOpenGLShaderProgram();
		Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/HoloWindow.vertex.glsl");
		Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/HoloWindow.fragment.glsl");
		if (!Program->bind()) std::cerr << "[Error] " << Program->log().toStdString() << '\n';
		#pragma endregion

		if (!Vao.create()) std::cerr << "[Error] Vao create() fail\n";
		Vao.bind();

		#pragma region Texture
		glGenTextures(1, &Tex);
		glBindTexture(GL_TEXTURE_2D, Tex);

		if (Queue.get_frame_desc().depth == 1)
		{
			uint	size = Queue.get_frame_desc().frame_size();
			uchar	*mTexture = new uchar[size];
			std::memset(mTexture, 0x00, size);
			glTexImage2D(GL_TEXTURE_2D, 0,
				GL_RGBA,
				Queue.get_frame_desc().width,
				Queue.get_frame_desc().height, 0,
				GL_RED, GL_UNSIGNED_BYTE, mTexture);
			delete[] mTexture;
		}
		else if (Queue.get_frame_desc().depth == 2)
		{
			uint	size = Queue.get_frame_desc().frame_size();
			ushort	*mTexture = new ushort[size];
			std::memset(mTexture, 0x00, size * 2);
			glTexImage2D(GL_TEXTURE_2D, 0,
				GL_RGBA,
				Queue.get_frame_desc().width,
				Queue.get_frame_desc().height, 0,
				GL_RG, GL_UNSIGNED_SHORT, mTexture);
			delete[] mTexture;
		}

		glUniform1i(glGetUniformLocation(Program->programId(), "tex"), 0);
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
		if (Queue.get_frame_desc().depth == 1)
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
		}

		glBindTexture(GL_TEXTURE_2D, 0);
		cudaGraphicsGLRegisterImage(&cuResource, Tex, GL_TEXTURE_2D,
			cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore);
		#pragma endregion

		#pragma region Vertex Buffer Object
		const float	data[16] = {
			// Top-left
			-vertCoord, vertCoord,	// vertex coord (-1.0f <-> 1.0f)
			0.0f, 0.0f,				// texture coord (0.0f <-> 1.0f)
			// Top-right
			vertCoord, vertCoord,
			texCoord, 0.0f,
			// Bottom-right
			vertCoord, -vertCoord,
			texCoord, texCoord,
			// Bottom-left
			-vertCoord, -vertCoord,
			0.0f, texCoord
		};

		glGenBuffers(1, &Vbo);
		glBindBuffer(GL_ARRAY_BUFFER, Vbo);
		glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(GLfloat), data, GL_STATIC_DRAW);

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

		Vao.release();
		Program->release();

		glViewport(0, 0, winSize.width(), winSize.height());
	}

	void HoloWindow::resizeGL(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void HoloWindow::paintGL()
	{
		makeCurrent();
		glClear(GL_COLOR_BUFFER_BIT);

		#pragma region Cuda
		cudaGraphicsMapResources(1, &cuResource, cuStream);
		cudaArray_t cuArr = nullptr;

		cudaGraphicsSubResourceGetMappedArray(&cuArr, cuResource, 0, 0);
		cudaResourceDesc cuArrRD;
		{
			cuArrRD.resType = cudaResourceTypeArray;
			cuArrRD.res.array.array = cuArr;
		}
		cudaSurfaceObject_t cuSurface;
		cudaCreateSurfaceObject(&cuSurface, &cuArrRD);
		{
			textureUpdate(cuSurface,
				Queue.get_last_images(1),
				Queue.get_frame_desc().width,
				Queue.get_frame_desc().height);
		}
		cudaDestroySurfaceObject(cuSurface);

		// Unmap the buffer for access by CUDA.
		cudaGraphicsUnmapResources(1, &cuResource, cuStream);
		cudaStreamSynchronize(cuStream);
		#pragma endregion

		glBindTexture(GL_TEXTURE_2D, Tex);
		glGenerateMipmap(GL_TEXTURE_2D);
		Program->bind();
		Vao.bind();

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		Vao.release();
		Program->release();
		glBindTexture(GL_TEXTURE_2D, 0);

		//update();
	}

	void HoloWindow::mousePressEvent(QMouseEvent* e)
	{
		if (kSelection == SliceZoom && !sliceLock)
		{
			if (e->button() == Qt::LeftButton)
			{
				e->pos();
				selectionRect.setTopLeft(QPoint(
					(e->x() * Fd.width) / width(),
					(e->y() * Fd.height) / height()));
				selectionRect.setBottomRight(selectionRect.topLeft());
				if (kSelection == KindOfSelection::None)
					kSelection = KindOfSelection::Zoom;
			}
			else if (e->button() == Qt::RightButton &&
				kSelection == KindOfSelection::Zoom)
			{
				// dezoom;
			}
		}
	}

	void HoloWindow::mouseMoveEvent(QMouseEvent* e)
	{

	}

	void HoloWindow::mouseReleaseEvent(QMouseEvent* e)
	{
		if (sliceLock)
		{
			selectionRect.setBottomRight(QPoint(
				(e->x() * Fd.width) / width(),
				(e->y() * Fd.height) / height()));
			selectionRect.setBottomLeft(QPoint(
				selectionRect.topLeft().x(),
				(e->y() * Fd.height) / height()));

			selectionRect.setTopRight(QPoint(
				(e->x() * Fd.width) / width(),
				selectionRect.topLeft().y()));

			//bounds_check(selectionRect);
			selectionRect.checkCorners();
			if (selectionRect.topLeft() != selectionRect.topRight())
				;// zoom(selectionRect);
			else if (selectionRect.topLeft() != selectionRect.bottomRight())
				;// zoom(selectionRect);
			sliceLock = false;
		}
	}

	void HoloWindow::wheelEvent(QWheelEvent *e)
	{
		if (e->x() < winSize.width() && e->y() < winSize.height())
		{
			if (e->angleDelta().y() > 0)
			{
				Scale += 0.1f * Scale;
				setScale();
			}
			else if (e->angleDelta().y() < 0)
			{
				if (Scale <= 1 || Scale < 1.1f)
					Scale = 1.f;
				else
					Scale -= 0.1f * Scale;
				setScale();
			}
		}
	}
	
	void HoloWindow::setScale()
	{
		if (Program)
		{
			makeCurrent();
			Program->bind();
			glUniform1f(glGetUniformLocation(Program->programId(), "scale"), Scale);
			Program->release();
		}
	}

}