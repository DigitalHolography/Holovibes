#include <QOpenGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

#include "gui_gl_widget.hh"
#include "queue.hh"
#include "tools_conversion.cuh"

namespace gui
{
	GLWidget::GLWidget(
		holovibes::Holovibes& h,
		holovibes::Queue& q,
		const unsigned int width,
		const unsigned int height,
		QWidget *parent)
		: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
		, QOpenGLFunctions()
		, h_(h)
		, timer_(this)
		, width_(width)
		, height_(height)
		, queue_(q)
		, frame_desc_(queue_.get_frame_desc())
		, buffer_(0)
		, cuda_buffer_(nullptr)
		, is_selection_enabled_(false)
		, selection_mode_(eselection::ZOOM)
		, is_signal_selection_(true)
		, px_(0.0f)
		, py_(0.0f)
		, zoom_ratio_(1.0f)
		, parent_(parent)
	{
		this->setObjectName("GLWidget");
		this->resize(QSize(width, height));
		connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
		timer_.start(1000 / DISPLAY_FRAMERATE);

		// Create a new computation stream on the graphics card.
		if (cudaStreamCreate(&cuda_stream_) != cudaSuccess)
			cuda_stream_ = 0; // Use default stream as a fallback

		num_2_shortcut = new QShortcut(QKeySequence(Qt::Key_2), this);
		num_2_shortcut->setContext(Qt::ApplicationShortcut);
		connect(num_2_shortcut, SIGNAL(activated()), this, SLOT(view_move_down()));

		num_4_shortcut = new QShortcut(QKeySequence(Qt::Key_4), this);
		num_4_shortcut->setContext(Qt::ApplicationShortcut);
		connect(num_4_shortcut, SIGNAL(activated()), this, SLOT(view_move_left()));

		num_6_shortcut = new QShortcut(QKeySequence(Qt::Key_6), this);
		num_6_shortcut->setContext(Qt::ApplicationShortcut);
		connect(num_6_shortcut, SIGNAL(activated()), this, SLOT(view_move_right()));

		num_8_shortcut = new QShortcut(QKeySequence(Qt::Key_8), this);
		num_8_shortcut->setContext(Qt::ApplicationShortcut);
		connect(num_8_shortcut, SIGNAL(activated()), this, SLOT(view_move_up()));

		key_plus_shortcut = new QShortcut(QKeySequence(Qt::Key_Plus), this);
		key_plus_shortcut->setContext(Qt::ApplicationShortcut);
		connect(key_plus_shortcut, SIGNAL(activated()), this, SLOT(view_zoom_out()));

		key_minus_shortcut = new QShortcut(QKeySequence(Qt::Key_Minus), this);
		key_minus_shortcut->setContext(Qt::ApplicationShortcut);
		connect(key_minus_shortcut, SIGNAL(activated()), this, SLOT(view_zoom_in()));
	}

	GLWidget::~GLWidget()
	{
		/* Unregister buffer for access by CUDA. */
		cudaGraphicsUnregisterResource(cuda_buffer_);
		/* Free the associated computation stream. */
		cudaStreamDestroy(cuda_stream_);
		/* Destroy buffer name. */
		glDeleteBuffers(1, &buffer_);
		glDisable(GL_TEXTURE_2D);
	}

	void GLWidget::view_move_down()
	{
		py_ += 0.1f / zoom_ratio_;
	}

	void GLWidget::view_move_left()
	{
		px_ += -0.1f / zoom_ratio_;
	}

	void GLWidget::view_move_right()
	{
		px_ += 0.1f / zoom_ratio_;
	}

	void GLWidget::view_move_up()
	{
		py_ += -0.1f / zoom_ratio_;
	}

	void GLWidget::view_zoom_out()
	{
		zoom_ratio_ *= 1.1f;
		glScalef(1.1f, 1.1f, 1.0f);
	}

	void GLWidget::view_zoom_in()
	{
		zoom_ratio_ *= 0.9f;
		glScalef(0.9f, 0.9f, 0.9f);
	}

	QSize GLWidget::minimumSizeHint() const
	{
		return QSize(width_, height_);
	}

	QSize GLWidget::sizeHint() const
	{
		return QSize(width_, height_);
	}

	void GLWidget::initializeGL()
	{
		initializeOpenGLFunctions();
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glEnable(GL_TEXTURE_2D);

		/* Generate buffer name. */
		glGenBuffers(1, &buffer_);

		/* Bind a named buffer object to the target GL_TEXTURE_BUFFER. */
		glBindBuffer(GL_TEXTURE_BUFFER, buffer_);

		//frame_desc_.frame_size();
		unsigned int size = frame_desc_.frame_size();
		if (frame_desc_.depth == 4 || frame_desc_.depth == 8)
			size /= 2;


		/* Creates and initialize a buffer object's data store. */
		glBufferData(
			GL_TEXTURE_BUFFER,
			size,
			//frame_desc_.frame_size(),
			nullptr,
			GL_DYNAMIC_DRAW);
		/* Unbind any buffer of GL_TEXTURE_BUFFER target. */
		glBindBuffer(GL_TEXTURE_BUFFER, 0);
		/* Register buffer name to CUDA. */
		cudaGraphicsGLRegisterBuffer(
			&cuda_buffer_,
			buffer_,
			cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);

		glViewport(0, 0, width_, height_);
	}

	void GLWidget::resizeGL(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void GLWidget::paintGL()
	{
		glEnable(GL_TEXTURE_2D);
		glClear(GL_COLOR_BUFFER_BIT);

		const void* frame = queue_.get_last_images(1);

		/* Map the buffer for access by CUDA. */
		cudaGraphicsMapResources(1, &cuda_buffer_, cuda_stream_);
		size_t buffer_size;
		void* buffer_ptr;
		cudaGraphicsResourceGetMappedPointer(&buffer_ptr, &buffer_size, cuda_buffer_);
		/* CUDA memcpy of the frame to opengl buffer. */

		if (frame_desc_.depth == 4)
			float_to_ushort(static_cast<const float *>(frame), static_cast<unsigned short *> (buffer_ptr), frame_desc_.frame_res());
		else if (frame_desc_.depth == 8)
			complex_to_ushort(static_cast<const cufftComplex *>(frame), static_cast<unsigned int *> (buffer_ptr), frame_desc_.frame_res());
		else
			cudaMemcpy(buffer_ptr, frame, buffer_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

		/* Unmap the buffer for access by CUDA. */
		cudaGraphicsUnmapResources(1, &cuda_buffer_, cuda_stream_);

		/* Bind the buffer object to the target GL_PIXEL_UNPACK_BUFFER.
		 * This affects glTexImage2D command. */
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_);

		if (frame_desc_.endianness == camera::BIG_ENDIAN)
			glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_TRUE);
		else
			glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);

		auto depth = GL_UNSIGNED_SHORT;
		if (frame_desc_.depth == 1)
			depth = GL_UNSIGNED_BYTE;

		auto kind = GL_RED;
		if (frame_desc_.depth == 8)
			kind = GL_RG;

		glTexImage2D(GL_TEXTURE_2D, 0, kind, frame_desc_.width, frame_desc_.height, 0, kind, depth, nullptr);
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		if (frame_desc_.depth == 8)
		{
			//We replace the green color by the blue one for complex display
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_GREEN);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
		}
		else
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
		}
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glBegin(GL_QUADS);
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		glTexCoord2d(0.0 + px_, 0.0 + py_); glVertex2d(-1.0, +1.0);
		glTexCoord2d(1.0 + px_, 0.0 + py_); glVertex2d(+1.0, +1.0);
		glTexCoord2d(1.0 + px_, 1.0 + py_); glVertex2d(+1.0, -1.0);
		glTexCoord2d(0.0 + px_, 1.0 + py_); glVertex2d(-1.0, -1.0);
		glEnd();

		glDisable(GL_TEXTURE_2D);

		if (is_selection_enabled_)
		{
			const float zoom_color[4] = { 0.0f, 0.5f, 0.0f, 0.4f };
			const float signal_color[4] = { 1.0f, 0.0f, 0.5f, 0.4f };
			const float noise_color[4] = { 0.26f, 0.56f, 0.64f, 0.4f };
			const float autofocus_color[4] = { 1.0f, 0.8f, 0.0f, 0.4f };
			const float stft_roi_color[4] = { 0.9f, 0.7f, 0.1f, 0.4f };

			switch (selection_mode_)
			{
			case AUTOFOCUS:
				selection_rect(selection_, autofocus_color);
				break;
			case AVERAGE:
				selection_rect(signal_selection_, signal_color);
				selection_rect(noise_selection_, noise_color);
				break;
			case ZOOM:
				selection_rect(selection_, zoom_color);
				break;
			case STFT_ROI:
				selection_rect(selection_, stft_roi_color);
				break;
			default:
				break;
			}
		}

		gl_error_checking();
	}

	void GLWidget::mousePressEvent(QMouseEvent* e)
	{
		if (e->button() == Qt::LeftButton)
		{
			is_selection_enabled_ = true;
			selection_.top_left = holovibes::Point2D(
				(e->x() * frame_desc_.width) / width(),
				(e->y() * frame_desc_.height) / height());
		}
		else
		if (selection_mode_ == ZOOM)
			dezoom();
	}

	void GLWidget::mouseMoveEvent(QMouseEvent* e)
	{
		if (is_selection_enabled_)
		{
			selection_.bottom_right = holovibes::Point2D(
				(e->x() * frame_desc_.width) / width(),
				(e->y() * frame_desc_.height) / height());

			if (selection_mode_ == AVERAGE)
			{
				if (is_signal_selection_)
					signal_selection_ = selection_;
				else // Noise selection
					noise_selection_ = selection_;
			}
			else if (selection_mode_ == STFT_ROI)
			{
				int max = std::abs(selection_.bottom_right.x - selection_.top_left.x);
				if (std::abs(selection_.bottom_right.y - selection_.top_left.y) > max)
					max = std::abs(selection_.bottom_right.y - selection_.top_left.y);

				selection_.bottom_right.x = selection_.top_left.x + max * ((selection_.top_left.x < selection_.bottom_right.x) * 2 - 1);
				selection_.bottom_right.y = selection_.top_left.y + max * ((selection_.top_left.y < selection_.bottom_right.y) * 2 - 1);
			}
		}
	}

	void GLWidget::mouseReleaseEvent(QMouseEvent* e)
	{
		if (is_selection_enabled_)
		{
			selection_.bottom_right = holovibes::Point2D(
				(e->x() * frame_desc_.width) / width(),
				(e->y() * frame_desc_.height) / height());

			if (selection_mode_ == STFT_ROI)
			{
				int max = std::abs(selection_.bottom_right.x - selection_.top_left.x);
				if (std::abs(selection_.bottom_right.y - selection_.top_left.y) > max)
					max = std::abs(selection_.bottom_right.y - selection_.top_left.y);

				selection_.bottom_right.x = selection_.top_left.x + max * ((selection_.top_left.x < selection_.bottom_right.x) * 2 - 1);
				selection_.bottom_right.y = selection_.top_left.y + max * ((selection_.top_left.y < selection_.bottom_right.y) * 2 - 1);
			}

			selection_.bottom_left = holovibes::Point2D(
				selection_.top_left.x,
				(e->y() * frame_desc_.height) / height());

			selection_.top_right = holovibes::Point2D(
				(e->x() * frame_desc_.width) / width(),
				selection_.top_left.y);

			bounds_check(selection_);
			swap_selection_corners(selection_);

			switch (selection_mode_)
			{
			case AUTOFOCUS:
				emit autofocus_zone_selected(selection_);
				selection_mode_ = ZOOM;
				break;
			case AVERAGE:
				if (is_signal_selection_)
				{
					signal_selection_ = selection_;
					h_.get_compute_desc().signal_zone = resize_zone(signal_selection_);
				}
				else // Noise selection
				{
					noise_selection_ = selection_;
					h_.get_compute_desc().noise_zone = resize_zone(noise_selection_);
				}
				is_signal_selection_ = !is_signal_selection_;
				break;
			case ZOOM:
				is_selection_enabled_ = false;

				if (selection_.top_left != selection_.bottom_right)
					zoom(selection_);
				break;
			case STFT_ROI:
				if (e->button() == Qt::LeftButton)
				{
					stft_roi_selection_ = selection_;
					emit stft_roi_zone_selected_update(stft_roi_selection_);
					emit stft_roi_zone_selected_end();
					selection_mode_ = ZOOM;
				}
				else
				{
					emit stft_roi_zone_selected_end();
					selection_mode_ = ZOOM;
				}
				break;
			default:
				break;
			}

			selection_ = holovibes::Rectangle();
		}
	}

	void GLWidget::selection_rect(const holovibes::Rectangle& selection, const float color[4])
	{
		const float xmax = frame_desc_.width;
		const float ymax = frame_desc_.height;

		float nstartx = (2.0f * static_cast<float>(selection.top_left.x)) / xmax - 1.0f;
		float nstarty = -1.0f * ((2.0f * static_cast<float>(selection.top_left.y)) / ymax - 1.0f);
		float nendx = (2.0f * static_cast<float>(selection.bottom_right.x)) / xmax - 1.0f;
		float nendy = -1.0f * ((2.0f * static_cast<float>(selection.bottom_right.y) / ymax - 1.0f));

		nstartx /= zoom_ratio_;
		nstarty /= zoom_ratio_;
		nendx /= zoom_ratio_;
		nendy /= zoom_ratio_;

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glBegin(GL_POLYGON);
		glColor4f(color[0], color[1], color[2], color[3]);
		glVertex2f(nstartx, nstarty);
		glVertex2f(nendx, nstarty);
		glVertex2f(nendx, nendy);
		glVertex2f(nstartx, nendy);
		glEnd();

		glDisable(GL_BLEND);
	}

	holovibes::Rectangle  GLWidget::resize_zone(holovibes::Rectangle selection)
	{
		selection.top_left.x /= zoom_ratio_;
		selection.bottom_left.x /= zoom_ratio_;
		selection.top_left.y /= zoom_ratio_;
		selection.top_right.y /= zoom_ratio_;

		selection.top_right.x /= zoom_ratio_;
		selection.bottom_right.x /= zoom_ratio_;
		selection.bottom_left.y /= zoom_ratio_;
		selection.bottom_right.y /= zoom_ratio_;
		return (selection);
	}

	void GLWidget::zoom(const holovibes::Rectangle& selection)
	{
		// Translation
		// Destination point is center of the window (OpenGL coords)
		const float xdest = 0.0f;
		const float ydest = 0.0f;

		// Source point is center of the selection zone (normal coords)
		const int xsource = selection.top_left.x + ((selection.bottom_right.x - selection.top_left.x) / 2);
		const int ysource = selection.top_left.y + ((selection.bottom_right.y - selection.top_left.y) / 2);

		// Normalizing source points to OpenGL coords
		const float nxsource = (2.0f * static_cast<float>(xsource)) / static_cast<float>(frame_desc_.width) - 1.0f;
		const float nysource = -1.0f * ((2.0f * static_cast<float>(ysource)) / static_cast<float>(frame_desc_.height) - 1.0f);

		// Projection of the translation
		const float px = xdest - nxsource;
		const float py = ydest - nysource;

		// Zoom ratio
		const float xratio = static_cast<float>(frame_desc_.width) /
			(static_cast<float>(selection.bottom_right.x) -
			static_cast<float>(selection.top_left.x));
		const float yratio = static_cast<float>(frame_desc_.height) /
			(static_cast<float>(selection.bottom_right.y) -
			static_cast<float>(selection.top_left.y));

		float min_ratio = xratio < yratio ? xratio : yratio;
		px_ += -px / zoom_ratio_ / 2;
		py_ += py / zoom_ratio_ / 2;
		zoom_ratio_ *= min_ratio;

		glScalef(min_ratio, min_ratio, 1.0f);
		parent_->setWindowTitle(QString("Real time display - zoom x") + QString(std::to_string(zoom_ratio_).c_str()));
	}

	void GLWidget::dezoom()
	{
		glLoadIdentity();
		zoom_ratio_ = 1.0f;
		px_ = 0.0f;
		py_ = 0.0f;
		parent_->setWindowTitle(QString("Real time display"));
	}

	void GLWidget::swap_selection_corners(holovibes::Rectangle& selection)
	{
		const int x_top_left = selection.top_left.x;
		const int y_top_left = selection.top_left.y;
		const int x_bottom_right = selection.bottom_right.x;
		const int y_bottom_rigth = selection.bottom_right.y;

		QPoint tmp;

		if (x_top_left < x_bottom_right)
		{
			if (y_top_left > y_bottom_rigth)
			{
				selection.horizontal_symetry();
			}
			//else
			//{
			//  This case is the default one, it doesn't need to be handled.
			//}
		}
		else
		{
			if (y_top_left < y_bottom_rigth)
			{
				selection.vertical_symetry();
			}
			else
			{
				// Vertical and horizontal swaps
				selection.vertical_symetry();
				selection.horizontal_symetry();
			}
		}
	}

	void GLWidget::bounds_check(holovibes::Rectangle& selection)
	{
		if (selection.bottom_right.x < 0)
			selection.bottom_right.x = 0;
		if (selection.bottom_right.x > frame_desc_.width)
			selection.bottom_right.x = frame_desc_.width;

		if (selection.bottom_right.y < 0)
			selection.bottom_right.y = 0;
		if (selection.bottom_right.y > frame_desc_.height)
			selection.bottom_right.y = frame_desc_.height;

		selection = holovibes::Rectangle(selection.top_left, selection.bottom_right);
	}

	void GLWidget::gl_error_checking()
	{
		// Sometimes this will occur when opengl is having some
		// trouble, and this will cause glGetString to return NULL.
		// That's why we need to check it, in order to avoid crashes.
		GLenum error = glGetError();
		auto err_string = glGetString(error);
		if (error != GL_NO_ERROR && err_string)
			std::cerr << "[GL] " << err_string << std::endl;
	}

	void GLWidget::resizeFromWindow(const int width, const int height)
	{
		resizeGL(width, height);
		resize(QSize(width, height));
	}
}