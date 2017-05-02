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
#include "SliceWindow.hh"

namespace holovibes
{
	namespace gui
	{
		SliceWindow::SliceWindow(QPoint p, QSize s, Queue& q, KindOfView k) :
			BasicOpenGLWindow(p, s, q, k),
			cuArray(nullptr),
			cuSurface(0),
			pIndex(0),
			Cd(nullptr)
		{}

		SliceWindow::~SliceWindow()
		{
			if (cuSurface) cudaDestroySurfaceObject(cuSurface);
			if (cuArray) cudaFreeArray(cuArray);
		}

		void	SliceWindow::setPIndex(ushort pId)
		{
			pIndex = pId + 1;
			if (Program)
			{
				makeCurrent();
				QPoint p = (kView == SliceXZ) ? QPoint(0, pIndex) : QPoint(pIndex, 0);
				QSize s = (kView == SliceXZ) ? QSize(Fd.width, Fd.height) : QSize(Fd.height, Fd.width);
				Overlay.setCrossBuffer(p, s);
			}
		}

		void	SliceWindow::setCd(ComputeDescriptor* cd)
		{
			Cd = cd;
		}

		void	SliceWindow::initShaders()
		{
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/vertex.holo.glsl");
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/fragment.tex.glsl");
			Program->link();
			Overlay.initShaderProgram();
			Overlay.initCrossBuffer();
			Overlay.setKind(KindOfOverlay::Cross);
		}

		void	SliceWindow::initializeGL()
		{
			makeCurrent();
			initializeOpenGLFunctions();
			glClearColor(0.128f, 0.128f, 0.128f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glBlendEquation(GL_FUNC_ADD);

			Vao.create();
			Vao.bind();
			initShaders();
			Program->bind();

			#pragma region Texture
			glGenTextures(1, &Tex);
			glBindTexture(GL_TEXTURE_2D, Tex);

			uint	size = Fd.frame_size();
			ushort	*mTexture = new ushort[size];
			std::memset(mTexture, 0, size * sizeof(ushort));

			glTexImage2D(GL_TEXTURE_2D, 0,
				GL_RGBA,
				Fd.width, Fd.height, 0,
				GL_RG, GL_UNSIGNED_SHORT, mTexture);

			glUniform1i(glGetUniformLocation(Program->programId(), "tex"), 0);
			glGenerateMipmap(GL_TEXTURE_2D);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			if (Fd.depth == 8.f)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_GREEN);
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
			}

			glBindTexture(GL_TEXTURE_2D, 0);
			delete[] mTexture;
			cudaGraphicsGLRegisterImage(&cuResource, Tex, GL_TEXTURE_2D,
				cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore);
			cudaGraphicsMapResources(1, &cuResource, cuStream);
			cudaGraphicsSubResourceGetMappedArray(&cuArray, cuResource, 0, 0);
			cuArrRD.resType = cudaResourceTypeArray;
			cuArrRD.res.array.array = cuArray;
			cudaCreateSurfaceObject(&cuSurface, &cuArrRD);
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
			glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

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

			glUniform1f(glGetUniformLocation(Program->programId(), "angle"), Angle * (M_PI / 180.f));
			glUniform1i(glGetUniformLocation(Program->programId(), "flip"), Flip);
			glUniform1f(glGetUniformLocation(Program->programId(), "scale"), Scale);
			glUniform2f(glGetUniformLocation(Program->programId(), "translate"), Translate[0], Translate[1]);

			Program->release();

			setPIndex(pIndex - 1);

			Vao.release();
			glViewport(0, 0, width(), height());
			startTimer(DISPLAY_RATE);
		}

		void	SliceWindow::paintGL()
		{
			makeCurrent();
			glClear(GL_COLOR_BUFFER_BIT);
			textureUpdate(cuSurface,
				Qu.get_last_images(1),
				Qu.get_frame_desc(),
				cuStream);

			glBindTexture(GL_TEXTURE_2D, Tex);
			glGenerateMipmap(GL_TEXTURE_2D);
			Vao.bind();

			Program->bind();
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);

			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			Program->release();

			Overlay.drawCross((kView == SliceXZ) ? 2 : 0, 2);

			Vao.release();
			glBindTexture(GL_TEXTURE_2D, 0);
		}

		void	SliceWindow::mousePressEvent(QMouseEvent* e)
		{}

		void	SliceWindow::mouseMoveEvent(QMouseEvent* e)
		{}

		void	SliceWindow::mouseReleaseEvent(QMouseEvent* e)
		{
			if (e->button() == Qt::RightButton)
				resetTransform();
		}
	
		void	SliceWindow::focusInEvent(QFocusEvent* e)
		{
			QWindow::focusInEvent(e);
			if (Cd)
			{
				Cd->current_window.exchange((kView == KindOfView::SliceXZ) ? WindowKind::SliceXZ : WindowKind::SliceYZ);
				Cd->notify_observers();
			}
		}
	}
}
