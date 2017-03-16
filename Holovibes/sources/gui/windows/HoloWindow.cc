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

#include <sstream>
#include <boost/algorithm/string.hpp>
#include "texture_update.cuh"
#include "HoloWindow.hh"

namespace gui
{

	std::atomic<bool> BasicOpenGLWindow::slicesAreLocked = true;

	HoloWindow::HoloWindow(QPoint p, QSize s, holovibes::Queue& q,
		holovibes::ComputeDescriptor &cd, KindOfView k) :
		/* ~~~~~~~~~~~~ */
		BasicOpenGLWindow(p, s, q, cd, k)
	{}

	HoloWindow::~HoloWindow()
	{}

	void HoloWindow::setKindOfSelection(KindOfSelection k)
	{
		zoneSelected.setKind(k);
	}

	const KindOfSelection HoloWindow::getKindOfSelection() const
	{
		return zoneSelected.getKind();
	}

	void HoloWindow::initializeGL()
	{
		makeCurrent();
		initializeOpenGLFunctions();
		glClearColor(0.128f, 0.128f, 0.128f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);

		#pragma region Shaders
		Program = new QOpenGLShaderProgram();
		Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/render.vertex.glsl");
		Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/render.fragment.glsl");
		if (!Program->bind()) std::cerr << "[Error] " << Program->log().toStdString() << '\n';
		#pragma endregion

		if (!Vao.create()) std::cerr << "[Error] Vao create() fail\n";
		Vao.bind();

		#pragma region Texture
		glGenTextures(1, &Tex);
		glBindTexture(GL_TEXTURE_2D, Tex);

		std::cout << "depth : " << Queue.get_frame_desc().depth << std::endl;
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
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);	// GL_CLAMP_TO_BORDER ~ GL_REPEAT
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);	// GL_CLAMP_TO_BORDER ~ GL_REPEAT
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);

		glBindTexture(GL_TEXTURE_2D, 0);

		cudaGraphicsGLRegisterImage(&cuResource, Tex, GL_TEXTURE_2D,
			cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore);

		cudaGraphicsMapResources(1, &cuResource, cuStream);
		cudaGraphicsSubResourceGetMappedArray(&cuArray, cuResource, 0, 0);
		cuArrRD.resType = cudaResourceTypeArray;
		cuArrRD.res.array.array = cuArray;
		cudaCreateSurfaceObject(&cuSurface, &cuArrRD);

		#pragma endregion

		#pragma region Vertex Buffer Object
		const float	data[] = {
			// Top-left
			-vertCoord, vertCoord,	// vertex coord (-1.0f <-> 1.0f)
			0.f, 0.f,				// texture coord (0.0f <-> 1.0f)
			// Top-right
			vertCoord, vertCoord,
			1.f, 0.f,
			// Bottom-right
			vertCoord, -vertCoord,
			1.f, 1.f,
			// Bottom-left
			-vertCoord, -vertCoord,
			0.f, 1.f
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
		const GLuint elements[] = {
			0, 1, 2,
			2, 3, 0
		};
		glGenBuffers(1, &Ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLuint), elements, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		#pragma endregion

		glUniform1i(glGetUniformLocation(Program->programId(), "flip"), 0);
		glUniform1f(glGetUniformLocation(Program->programId(), "scale"), Scale);
		glUniform1f(glGetUniformLocation(Program->programId(), "angle"), 0 * (M_PI / 180.f));
		glUniform2f(glGetUniformLocation(Program->programId(), "translate"), Translate[0], Translate[1]);

		Program->release();

		zoneSelected.initShaderProgram();

		Vao.release();

		glViewport(0, 0, winSize.width(), winSize.height());
		startTimer(DisplayRate);
	}

	void HoloWindow::resizeGL(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void HoloWindow::paintGL()
	{
		makeCurrent();
		glClear(GL_COLOR_BUFFER_BIT);

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

		Program->release();
		glBindTexture(GL_TEXTURE_2D, 0);

		if (zoneSelected.isEnabled())
		{
			zoneSelected.draw();
		}
		Vao.release();
	}

	void HoloWindow::mousePressEvent(QMouseEvent* e)
	{
		if (slicesAreLocked.load())
		{
			if (e->button() == Qt::LeftButton)
			{
				zoneSelected.press(e->pos());
			}
		}
	}

	void HoloWindow::mouseMoveEvent(QMouseEvent* e)
	{
		if (!slicesAreLocked.load())
		{
			updateCursorPosition(QPoint(
				e->x() * (Fd.width / static_cast<float>(width())),
				e->y() * (Fd.height / static_cast<float>(height()))));
		}
		else
		{
			if (e->buttons() == Qt::LeftButton)
			{
				zoneSelected.move(e->pos());
			}
		}
	}

	void HoloWindow::mouseReleaseEvent(QMouseEvent* e)
	{
		if (e->button() == Qt::LeftButton)
		{
			zoneSelected.release();
			if (zoneSelected.getZone().topLeft() != zoneSelected.getZone().bottomRight() &&
				zoneSelected.getKind() == Zoom)
				zoomInRect(zoneSelected.getZone());
			/*else
			{
				Scale = 2;
				setScale();
			}*/
		}
		else if (e->button() == Qt::RightButton)
			resetTransform();

		/*if (sliceLock)
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
			sliceLock.exchange(false);
		}*/
	}

	void HoloWindow::wheelEvent(QWheelEvent *e)
	{
		if (e->x() < winSize.width() && e->y() < winSize.height())
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
				if (Scale <= 1 || Scale < 1.1f)
				{
					resetTransform();
				}
				else
				{
					Scale -= 0.1f * Scale;
					setScale();
					Translate[0] -= -xGL * 0.1 / Scale;
					Translate[1] -= yGL * 0.1 / Scale;
					setTranslate();
				}
			}
		}
	}

	void HoloWindow::zoomInRect(Rectangle zone)
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

	void HoloWindow::updateCursorPosition(QPoint pos)
	{
		auto manager = InfoManager::get_manager();
		std::stringstream ss;
		ss << "(Y,X) = (" << pos.y() << "," << pos.x() << ")";
		manager->update_info("STFT Slice Cursor", ss.str());
		Cd.stftCursor(&pos, holovibes::ComputeDescriptor::Set);
	}

}