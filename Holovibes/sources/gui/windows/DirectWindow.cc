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

namespace holovibes
{
	namespace gui
	{
		DirectWindow::DirectWindow(QPoint p, QSize s, Queue& q, ComputeDescriptor& cd) :
			BasicOpenGLWindow(p, s, q, KindOfView::Direct),
			texDepth(0),
			texType(0),
			compute_desc_(cd)
		{}

		DirectWindow::DirectWindow(QPoint p, QSize s, Queue& q, ComputeDescriptor& cd, KindOfView k) :
			BasicOpenGLWindow(p, s, q, k),
			compute_desc_(cd)
		{}

		DirectWindow::~DirectWindow()
		{}

		Rectangle	DirectWindow::getSignalZone() const
		{
			return (Overlay.getRectBuffer());
		}

		Rectangle	DirectWindow::getNoiseZone() const
		{
			return (Overlay.getRectBuffer(KindOfOverlay::Noise));
		}

		void	DirectWindow::setSignalZone(Rectangle signal)
		{
			Overlay.setZoneBuffer(width(), signal, KindOfOverlay::Signal);
		}

		void	DirectWindow::setNoiseZone(Rectangle noise)
		{
			Overlay.setZoneBuffer(width(), noise, KindOfOverlay::Noise);
		}

		void	DirectWindow::initShaders()
		{
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/vertex.direct.glsl");
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/fragment.tex.glsl");
			Program->link();
			Overlay.initShaderProgram();
		}

		void	DirectWindow::initializeGL()
		{
			makeCurrent();
			initializeOpenGLFunctions();
			glClearColor(0.128f, 0.128f, 0.128f, 1.0f);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glBlendEquation(GL_FUNC_ADD);

			Vao.create();
			Vao.bind();
			initShaders();
			Program->bind();

			#pragma region Texture
			glGenBuffers(1, &Pbo);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Pbo);
			const uint size = Fd.frame_size() / ((Fd.depth == 4 || Fd.depth == 8) ? 2 : 1);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
			glPixelStorei(GL_UNPACK_SWAP_BYTES,
				(Fd.byteEndian == Endianness::BigEndian) ?
				GL_TRUE : GL_FALSE);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			cudaGraphicsGLRegisterBuffer(&cuResource, Pbo,
				cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
			/* -------------------------------------------------- */
			glGenTextures(1, &Tex);
			glBindTexture(GL_TEXTURE_2D, Tex);
			texDepth = (Fd.depth == 1.f) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT;
			texType = (Fd.depth == 8.f) ? GL_RG : GL_RED;
			glTexImage2D(GL_TEXTURE_2D, 0, texType, Fd.width, Fd.height, 0, texType, texDepth, nullptr);

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
			glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(float), data, GL_DYNAMIC_DRAW);

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
			if (kView == KindOfView::Hologram)
			{
				glUniform1i(glGetUniformLocation(Program->programId(), "flip"), Flip);
				glUniform1f(glGetUniformLocation(Program->programId(), "angle"), Angle * (M_PI / 180.f));
			}
			Program->release();
			Vao.release();
			glViewport(0, 0, width(), height());
			startTimer(1000 / static_cast<float>(compute_desc_.display_rate.load()));
		}
		
		void	DirectWindow::resizeGL(int w, int h)
		{
			const int min = std::min(w, h);

			setFramePosition(QPoint(0, 0));

			if (State == Qt::WindowNoState)
			{
				if ((min != width() || min != height()))
					resize(min, min);
			}
			else if (State == Qt::WindowFullScreen)
				resize(w, h);

			glViewport(0, 0, min, min);
		}

		void	DirectWindow::paintGL()
		{
			makeCurrent();
			glClear(GL_COLOR_BUFFER_BIT);
			Vao.bind();
			Program->bind();

			cudaGraphicsMapResources(1, &cuResource, cuStream);
			cudaGraphicsResourceGetMappedPointer(&cuPtrToPbo, &sizeBuffer, cuResource);
			void* frame = Qu.get_last_images(1);
			if (Fd.depth == 4.f)
				float_to_ushort(static_cast<const float*>(frame), cuPtrToPbo, Fd.frame_res(), Fd.depth);
			else if (Fd.depth == 8.f)
				complex_to_ushort(static_cast<const cuComplex*>(frame), static_cast<uint*>(cuPtrToPbo), Fd.frame_res());
			else
				cudaMemcpy(cuPtrToPbo, frame, sizeBuffer, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
			cudaGraphicsUnmapResources(1, &cuResource, cuStream);
			cudaStreamSynchronize(cuStream);

			glBindTexture(GL_TEXTURE_2D, Tex);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Pbo);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Fd.width, Fd.height, texType, texDepth, nullptr);
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
			if (Overlay.isEnabled())
				Overlay.drawSelections();
			if (kView == KindOfView::Direct)
				Vao.release();
		}

		void	DirectWindow::mousePressEvent(QMouseEvent* e)
		{
			if (e->button() == Qt::LeftButton)
				Overlay.press(e->pos());
		}

		void	DirectWindow::mouseMoveEvent(QMouseEvent* e)
		{
			if (e->buttons() == Qt::LeftButton)
				Overlay.move(e->pos(), size());
		}

		void	DirectWindow::mouseReleaseEvent(QMouseEvent* e)
		{
			if (e->button() == Qt::LeftButton)
			{
				Overlay.release(width());
				if (Overlay.getConstZone().topLeft() !=
					Overlay.getConstZone().bottomRight())
				{
					if (Overlay.getKind() == Zoom)
						zoomInRect(Overlay.getConstZone());
				}
			}
			else if (e->button() == Qt::RightButton &&
				Overlay.getKind() != Signal &&
				Overlay.getKind() != Noise)
				resetTransform();
		}

		void	DirectWindow::zoomInRect(Rectangle zone)
		{
			const QPoint center = zone.center();

			Translate[0] += ((static_cast<float>(center.x()) / static_cast<float>(width())) - 0.5f) / Scale;
			Translate[1] += ((static_cast<float>(center.y()) / static_cast<float>(height())) - 0.5f) / Scale;
			setTranslate();

			const float xRatio = static_cast<float>(width()) / static_cast<float>(zone.width());
			const float yRatio = static_cast<float>(height()) / static_cast<float>(zone.height());

			Scale = (xRatio < yRatio ? xRatio : yRatio) * Scale;
			setScale();
		}
	}
}
