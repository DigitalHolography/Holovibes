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

#ifdef WIN32
  #include <windows.h>
#endif
#include <cuda_gl_interop.h>

#include <QGuiApplication>
#include <QMouseEvent>
#include <QRect>
#include <QScreen>
#include <QWheelEvent>

#include "DirectWindow.hh"
#include "HoloWindow.hh"
#include "info_manager.hh"
#include "cuda_memory.cuh"
#include "Common.cuh"

namespace holovibes
{
	using camera::FrameDescriptor;
	using camera::Endianness;
	namespace gui
	{
		DirectWindow::DirectWindow(QPoint p, QSize s, std::unique_ptr<Queue>& q, KindOfView k) :
			BasicOpenGLWindow(p, s, q, k),
			texDepth(0),
			texType(0)
		{
		}

		DirectWindow::~DirectWindow()
		{}

		void DirectWindow::initShaders()
		{
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/vertex.direct.glsl");
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/fragment.tex.glsl");
			Program->link();
			overlay_manager_.create_default();
		}

		void DirectWindow::initializeGL()
		{
			makeCurrent();
			initializeOpenGLFunctions();
			//glClearColor(0.128f, 0.128f, 0.128f, 1.0f);
			glClearColor(0.f, 0.f, 0.f, 1.0f);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glBlendEquation(GL_FUNC_ADD);

			initShaders();
			Vao.create();
			Vao.bind();
			Program->bind();

			#pragma region Texture
			glGenBuffers(1, &Pbo);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Pbo);
			const uint size = fd_.frame_size() / ((fd_.depth == 4 || fd_.depth == 8) ? 2 : 1);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_STATIC_DRAW);	//GL_STATIC_DRAW ~ GL_DYNAMIC_DRAW
			glPixelStorei(GL_UNPACK_SWAP_BYTES,
				(fd_.byteEndian == Endianness::BigEndian) ?
				GL_TRUE : GL_FALSE);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			cudaGraphicsGLRegisterBuffer(&cuResource, Pbo,
				cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
			/* -------------------------------------------------- */
			glGenTextures(1, &Tex);
			glBindTexture(GL_TEXTURE_2D, Tex);
			texDepth = (fd_.depth == 1) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT;
			texType = (fd_.depth == 8) ? GL_RG : GL_RED;
			if (fd_.depth == 6)
				texType = GL_RGB;
			glTexImage2D(GL_TEXTURE_2D, 0, texType, fd_.width, fd_.height, 0, texType, texDepth, nullptr);

			Program->setUniformValue(Program->uniformLocation("tex"), 0);

			glGenerateMipmap(GL_TEXTURE_2D);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);	// GL_NEAREST ~ GL_LINEAR
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			if (fd_.depth == 8)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_GREEN);
			}
			else if (fd_.depth != 6)
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
				0.f, 0.f,		// texture coord (0.0f <-> 1.0f)
				// Top-right
				1.f, 1.f,
				1.f, 0.f,
				// Bottom-right
				1.f, -1.f,
				1.f, 1.f,
				// Bottom-left
				-1.f, -1.f,
				0.f, 1.f
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

			setTransform();

			Program->release();
			Vao.release();
			glViewport(0, 0, width(), height());
			startTimer(1000 / cd_->display_rate);
		}


		/* This part of code makes a resizing of the window displaying image to
		   a rectangle format. It also avoids the window to move when resizing.
		   There is no visible calling function since it's overriding Qt function.
		**/
		void DirectWindow::resizeGL(int w, int h)
		{
			if (ratio == 0.0f)
				return;
			int tmp_width = old_width;
			int tmp_height = old_height;

			auto point = this->position();

			if ((cd_->compute_mode == Computation::Hologram && cd_->algorithm == Algorithm::None)
				|| cd_->compute_mode == Computation::Direct)
			{
				if (w != old_width)
				{
					old_width = w;
					old_height = w / ratio;
				}
				else if (h != old_height)
				{
					old_width = h * ratio;
					old_height = h;
				}
			}
			else
			{
				if (is_resize)
				{
					if (w != old_width)
					{
						old_height = w;
						old_width = w;
					}
					else if (h != old_height)
					{
						old_height = h;
						old_width = h;
					}
				}
				else
				{
					old_height = std::max(h, w);
					old_width = old_height;
				}
				is_resize = true;

				if (old_height < 140 || old_width < 140)
				{
					old_height = tmp_height;
					old_width = tmp_width;
				}
				is_resize = true;
			}

			QRect screen = QGuiApplication::primaryScreen()->geometry();
			if (old_height >  screen.height() || old_width > screen.width())
			{
				old_height = tmp_height;
				old_width = tmp_width;
			}
			resize(old_width, old_height);
			this->setPosition(point);
		}

		void DirectWindow::paintGL()
		{
			// Window translation but none seems to be performed
			glViewport(0, 0, width(), height());

			// Bind framebuffer to the context, "not necessary to call this function in most cases, because it is called automatically before invoking paintGL()."
			makeCurrent();

			// Clear buffer
			glClear(GL_COLOR_BUFFER_BIT);

			// Binds the vertex array object to the OpenGL binding point
			Vao.bind();
			Program->bind();

			// Map resources for CUDA
			cudaSafeCall(cudaGraphicsMapResources(1, &cuResource, cuStream));
			// Retrive the cuda pointer
			cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&cuPtrToPbo, &sizeBuffer, cuResource));

			// Get the last image from the ouput queue
			void* frame = output_->get_last_images(1);

			// Put the frame inside the cuda ressrouce
			if (cd_->img_type == ImgType::Composite)
			{
				cudaXMemcpy(cuPtrToPbo, frame, sizeBuffer, cudaMemcpyDeviceToDevice);
			}
			else
			{
				convert_frame_for_display(frame, cuPtrToPbo, fd_.frame_res(), fd_.depth, kView == KindOfView::Direct ? cd_->direct_bitshift.load() : 0);
			}

			// Release resources (needs to be done at each call) and sync, sync usefull since memcpy not async and kernel has a cudaDeviceSyncronize ?
			cudaSafeCall(cudaGraphicsUnmapResources(1, &cuResource, cuStream));
			cudaSafeCall(cudaStreamSynchronize(cuStream));

			// Texture creationg
			glBindTexture(GL_TEXTURE_2D, Tex);

			// Binds buffer to texture data source
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Pbo);

			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fd_.width, fd_.height, texType, texDepth, nullptr);
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
			Vao.release();

			overlay_manager_.draw();
		}

		void DirectWindow::mousePressEvent(QMouseEvent* e)
		{
			overlay_manager_.press(e);
		}

		void DirectWindow::mouseMoveEvent(QMouseEvent* e)
		{
			overlay_manager_.move(e);
		}

		void DirectWindow::mouseReleaseEvent(QMouseEvent* e)
		{
			if (e->button() == Qt::LeftButton)
				overlay_manager_.release(fd_.width);
			else if (e->button() == Qt::RightButton)
				resetTransform();
		}

		void DirectWindow::keyPressEvent(QKeyEvent * e)
		{
			BasicOpenGLWindow::keyPressEvent(e);

			float translation = translation_step_ / scale_;

			switch (e->key()) {
			case Qt::Key::Key_8:
				translate_[1] -= translation;
				break;
			case Qt::Key::Key_2:
				translate_[1] += translation;
				break;
			case Qt::Key::Key_6:
				translate_[0] += translation;
				break;
			case Qt::Key::Key_4:
				translate_[0] -= translation;
				break;
			}
			setTransform();
		}

		void DirectWindow::zoomInRect(units::RectOpengl zone)
		{
			const units::PointOpengl center = zone.center();

			const float delta_x = center.x() / (getScale() * 2);
			const float delta_y = center.y() / (getScale() * 2);

			const auto old_translate = getTranslate();

			const auto new_translate_x = old_translate[0] + delta_x;
			const auto new_translate_y = old_translate[1] - delta_y;

			setTranslate(new_translate_x, new_translate_y);

			const float xRatio = zone.unsigned_width();

			// Use the commented line below if you are using square windows,
			// and comment the 2 lines below
			const float yRatio = zone.unsigned_height();
			setScale(getScale() / (std::min(xRatio, yRatio) / 2));
			//setScale(getScale() * 2.f / xRatio);

			setTransform();
		}

		void DirectWindow::setRatio(float ratio_)
		{
			ratio = ratio_;
		}

		bool DirectWindow::is_resize_call() const
		{
			return is_resize;
		}

		void DirectWindow::set_is_resize(bool b)
		{
			is_resize = b;
		}

		void DirectWindow::wheelEvent(QWheelEvent *e)
		{
			if (!is_between(e->x(), 0, width()) || !is_between(e->y(), 0, height()))
				return;
			const float xGL = (static_cast<float>(e->x() - width() / 2)) / static_cast<float>(width()) * 2.f;
			const float yGL = -((static_cast<float>(e->y() - height() / 2)) / static_cast<float>(height())) * 2.f;
			if (e->angleDelta().y() > 0)
			{
				scale_ += 0.1f * scale_;
				translate_[0] += xGL * 0.1 / scale_;
				translate_[1] += -yGL * 0.1 / scale_;
				setTransform();
			}
			else if (e->angleDelta().y() < 0)
			{
				scale_ -= 0.1f * scale_;
				if (scale_ < 1.f)
					scale_ = 1;
				else
				{
					translate_[0] -= xGL * 0.1 / scale_;
					translate_[1] -= -yGL * 0.1 / scale_;
					setTransform();
				}
			}
		}

	}
}
