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

#include <QOpenGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

#include "gui_gl_widget.hh"
#include "queue.hh"
#include "tools_conversion.cuh"
#include "info_manager.hh"

using namespace holovibes;

namespace gui
{
	GLWidget::GLWidget(
		Holovibes& h,
		Queue& q,
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
		, slice_block_(false)
	{
		const camera::FrameDescriptor& input_fd = h_.get_capture_queue().get_frame_desc();
		this->setObjectName("GLWidget");
		this->resize(QSize(width, height));
		connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
		timer_.start(1000 / DISPLAY_FRAMERATE);
		windowTitle = 
					QString("Input : ")
					+ QString(std::to_string(input_fd.width).c_str())
					+ QString("x") + QString(std::to_string(input_fd.height).c_str())
					+ QString(" ") + QString(std::to_string(static_cast<int>(input_fd.depth) << 3).c_str()) + QString("bit")
					+ QString(" ")
					+ QString("Output : ")
					+ QString(std::to_string(frame_desc_.width).c_str())
					+ QString("x") + QString(std::to_string(frame_desc_.height).c_str())
					+ QString(" ") + QString(std::to_string(static_cast<int>(frame_desc_.depth) << 3).c_str()) + QString("bit");
		parent_->setWindowTitle(windowTitle);
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

		key_space_shortcut = new QShortcut(QKeySequence(Qt::Key_Space), this);
		key_space_shortcut->setContext(Qt::ApplicationShortcut);
		connect(key_space_shortcut, SIGNAL(activated()), this, SLOT(block_slice()));
	}

	GLWidget::~GLWidget()
	{
		delete key_space_shortcut;
		delete key_minus_shortcut;
		delete key_plus_shortcut;
		delete num_8_shortcut;
		delete num_6_shortcut;
		delete num_4_shortcut;
		delete num_2_shortcut;

		makeCurrent();
		/* Unregister buffer for access by CUDA. */
		cudaGraphicsUnregisterResource(cuda_buffer_);
		/* Free the associated computation stream. */
		cudaStreamDestroy(cuda_stream_);
		/* Destroy buffer name. */
		glDeleteBuffers(1, &buffer_);
		glDisable(GL_TEXTURE_2D);
		doneCurrent();
	}

	void GLWidget::view_move_down()
	{
		py_ += 0.1f / zoom_ratio_;
	}

	void GLWidget::view_move_left()
	{
		px_ -= 0.1f / zoom_ratio_;
	}

	void GLWidget::view_move_right()
	{
		px_ += 0.1f / zoom_ratio_;
	}

	void GLWidget::view_move_up()
	{
		py_ -= 0.1f / zoom_ratio_;
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

	void GLWidget::block_slice()
	{
		slice_block_ = !slice_block_;
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
		makeCurrent();
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
		glBufferData(GL_TEXTURE_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
		/* Unbind any buffer of GL_TEXTURE_BUFFER target. */
		glBindBuffer(GL_TEXTURE_BUFFER, 0);
		/* Register buffer name to CUDA. */
		cudaGraphicsGLRegisterBuffer(
			&cuda_buffer_,
			buffer_,
			cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);

		glViewport(0, 0, width_, height_);
		//doneCurrent();
	}

	void GLWidget::resizeGL(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void GLWidget::paintGL()
	{
		makeCurrent();
		glEnable(GL_TEXTURE_2D);
		glClear(GL_COLOR_BUFFER_BIT);

		const void *frame = queue_.get_last_images(1);

		/* Map the buffer for access by CUDA. */
		cudaGraphicsMapResources(1, &cuda_buffer_, cuda_stream_);
		size_t	buffer_size;
		void*	buffer_ptr;
		// Recuperation d'un pointeur sur un buffer sur la CG a partir de la ressource : cuda_buffer_.
		// Retourne aussi la size de ce buffer : buffer_size
		cudaGraphicsResourceGetMappedPointer(&buffer_ptr, &buffer_size, cuda_buffer_);

		if (frame_desc_.depth == 4.f)
			float_to_ushort(static_cast<const float *>(frame), static_cast<unsigned short *> (buffer_ptr), frame_desc_.frame_res());
		else if (frame_desc_.depth == 8.f)
			complex_to_ushort(static_cast<const cufftComplex *>(frame), static_cast<unsigned int *> (buffer_ptr), frame_desc_.frame_res());
		else
			// CUDA memcpy of the frame to opengl buffer.
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
		if (frame_desc_.depth == 1.f)
			depth = GL_UNSIGNED_BYTE;

		auto kind = GL_RED;
		if (frame_desc_.depth == 8.f)
			kind = GL_RG;

		glTexImage2D(GL_TEXTURE_2D, 0, kind, frame_desc_.width, frame_desc_.height, 0, kind, depth, nullptr);
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		if (frame_desc_.depth == 8.f)
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
		glTexCoord2d(0.0f + px_, 0.0f + py_); glVertex2d(-1.0f, +1.0f);
		glTexCoord2d(1.0f + px_, 0.0f + py_); glVertex2d(+1.0f, +1.0f);
		glTexCoord2d(1.0f + px_, 1.0f + py_); glVertex2d(+1.0f, -1.0f);
		glTexCoord2d(0.0f + px_, 1.0f + py_); glVertex2d(-1.0f, -1.0f);
		glEnd();

		glDisable(GL_TEXTURE_2D);

		if (is_selection_enabled_)
		{
			const float zoom_color[4] = { 0.0f, 0.5f, 0.0f, 0.4f };
			const float signal_color[4] = { 1.0f, 0.0f, 0.5f, 0.4f };
			const float noise_color[4] = { 0.26f, 0.56f, 0.64f, 0.4f };
			const float autofocus_color[4] = { 1.0f, 0.8f, 0.0f, 0.4f };
			const float stft_roi_color[4] = { 0.0f, 0.62f, 1.0f, 0.4f };
			const float stft_slice_color[4] = { 1.0f, 0.87f, 0.87f, 0.4f };

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
			case STFT_SLICE:
				selection_rect(selection_, stft_slice_color);
				break;
			default:
				break;
			}
		}
		gl_error_checking();
		//	doneCurrent();
	}

	void GLWidget::mousePressEvent(QMouseEvent* e)
	{
		if (e->buttons() == Qt::LeftButton)
		{
			if (selection_mode_ == STFT_SLICE && !slice_block_)
				return;
			selection_.setTopLeft(QPoint(
				(e->x() * frame_desc_.width) / width(),
				(e->y() * frame_desc_.height) / height()));
			selection_.setBottomRight(selection_.topLeft());
			is_selection_enabled_ = true;
		}
		else if (e->buttons() == Qt::RightButton)
		{
			if (selection_mode_ == ZOOM)
				dezoom();
			else if (selection_mode_ == AVERAGE)
				is_selection_enabled_ = false;
		}
	}

	void GLWidget::mouseMoveEvent(QMouseEvent* e)
	{
		if (is_selection_enabled_)
		{
			if (e->buttons() == Qt::LeftButton)
			{
				selection_.setBottomRight(QPoint(
					(e->x() * frame_desc_.width) / width(),
					(e->y() * frame_desc_.height) / height()));

				if (selection_mode_ == AVERAGE)
				{
					if (is_signal_selection_)
						signal_selection_ = selection_;
					else // Noise selection
						noise_selection_ = selection_;
				}
				else if (selection_mode_ == STFT_ROI)
				{
					int max = std::abs(selection_.bottomRight().x() - selection_.topLeft().x());
					if (std::abs(selection_.bottomRight().y() - selection_.topLeft().y()) > max)
						max = std::abs(selection_.bottomRight().y() - selection_.topLeft().y());

					selection_.bottomRight().setX(
						selection_.topLeft().x() +
						max * ((selection_.topLeft().x() < selection_.bottomRight().x()) * 2 - 1));
					selection_.bottomRight().setY(
						selection_.topLeft().y() +
						max * ((selection_.topLeft().y() < selection_.bottomRight().y()) * 2 - 1));
				}
			}
		}
		if (selection_mode_ == STFT_SLICE && !slice_block_)
		{
			QPoint pos = QPoint(e->x() * (frame_desc_.width / static_cast<float>(width())),
				e->y() * (frame_desc_.height / static_cast<float>(height())));
			stft_slice_pos_update(pos);
		}
		else if (selection_mode_ != STFT_SLICE)
			slice_block_ = false;
	}

	void GLWidget::mouseReleaseEvent(QMouseEvent* e)
	{
		if (is_selection_enabled_)
		{
			selection_.setBottomRight(QPoint(
				(e->x() * frame_desc_.width) / width(),
				(e->y() * frame_desc_.height) / height()));

			if (selection_mode_ == STFT_ROI)
			{
				int max = std::abs(selection_.bottomRight().x() - selection_.topLeft().x());
				if (std::abs(selection_.bottomRight().y() - selection_.topLeft().y()) > max)
					max = std::abs(selection_.bottomRight().y() - selection_.topLeft().y());

				selection_.bottomRight().setX(
					selection_.topLeft().x() +
					max * ((selection_.topLeft().x() < selection_.bottomRight().x()) * 2 - 1));
				selection_.bottomRight().setY(
					selection_.topLeft().y() +
					max * ((selection_.topLeft().y() < selection_.bottomRight().y()) * 2 - 1));
			}

			selection_.setBottomLeft(QPoint(
				selection_.topLeft().x(),
				(e->y() * frame_desc_.height) / height()));

			selection_.setTopRight(QPoint(
				(e->x() * frame_desc_.width) / width(),
				selection_.topLeft().y()));

			bounds_check(selection_);
			swap_selection_corners(selection_);
			selection_.checkCorners();

			switch (selection_mode_)
			{
				case AUTOFOCUS:
					emit autofocus_zone_selected(selection_);
					selection_mode_ = get_selection_mode();
					is_selection_enabled_ = false;
					break;
				case AVERAGE:
					if (is_signal_selection_)
					{
						holovibes::Rectangle rect(resize_zone((signal_selection_ = selection_)));
						h_.get_compute_desc().signalZone(&rect, ComputeDescriptor::Set);
					}
					else // Noise selection
					{
						holovibes::Rectangle rect(resize_zone((noise_selection_ = selection_)));
						h_.get_compute_desc().noiseZone(&rect, ComputeDescriptor::Set);
					}
					is_signal_selection_ = !is_signal_selection_;
					break;
				case ZOOM:
					if (selection_.topLeft() != selection_.topRight())
						zoom(selection_);
					else if (selection_.topLeft() != selection_.bottomRight())
						zoom(selection_);
					is_selection_enabled_ = false;
					break;
				case STFT_ROI:
					if (e->button() == Qt::LeftButton)
					{
						stft_roi_selection_ = selection_;
						emit stft_roi_zone_selected_update(stft_roi_selection_);
						emit stft_roi_zone_selected_end();
					}
					else if (e->button() == Qt::RightButton)
					{
						emit stft_roi_zone_selected_end();
					}
					selection_mode_ = get_selection_mode();
					is_selection_enabled_ = false;
					break;
				case STFT_SLICE:
					is_selection_enabled_ = false;
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

		float nstartx = (2.0f * static_cast<float>(selection.topLeft().x())) / xmax - 1.0f;
		float nstarty = -1.0f * ((2.0f * static_cast<float>(selection.topLeft().y())) / ymax - 1.0f);
		float nendx = (2.0f * static_cast<float>(selection.bottomRight().x())) / xmax - 1.0f;
		float nendy = -1.0f * ((2.0f * static_cast<float>(selection.bottomRight().y()) / ymax - 1.0f));

		const float zr = 1 / zoom_ratio_;
		nstartx *= zr;
		nstarty *= zr;
		nendx *= zr;
		nendy *= zr;

		makeCurrent();
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
		//doneCurrent();
	}

	holovibes::Rectangle  GLWidget::resize_zone(holovibes::Rectangle selection)
	{
		const float zr = 1 / zoom_ratio_;

		selection.setTopRight(selection.topRight() * zr);

		selection.setTopLeft(selection.topLeft() * zr);

		selection.setBottomRight(selection.bottomRight() * zr);

		selection.setBottomLeft(selection.bottomLeft() * zr);
		return (selection);
	}

	void GLWidget::zoom(const holovibes::Rectangle& selection)
	{
		// Translation
		// Destination point is center of the window (OpenGL coords)
		const float xdest = 0.0f;
		const float ydest = 0.0f;

		// Source point is center of the selection zone (normal coords)
		const int xsource = selection.topLeft().x() + ((selection.bottomRight().x() - selection.topLeft().x()) / 2);
		const int ysource = selection.topLeft().y() + ((selection.bottomRight().y() - selection.topLeft().y()) / 2);

		// Normalizing source points to OpenGL coords
		const float nxsource = (2.0f * static_cast<float>(xsource)) / static_cast<float>(frame_desc_.width) - 1.0f;
		const float nysource = -1.0f * ((2.0f * static_cast<float>(ysource)) / static_cast<float>(frame_desc_.height) - 1.0f);

		// Projection of the translation
		const float px = xdest - nxsource;
		const float py = ydest - nysource;

		// Zoom ratio
		const float xratio = static_cast<float>(frame_desc_.width) /
			(static_cast<float>(selection.bottomRight().x()) -
				static_cast<float>(selection.topLeft().x()));
		const float yratio = static_cast<float>(frame_desc_.height) /
			(static_cast<float>(selection.bottomRight().y()) -
				static_cast<float>(selection.topLeft().y()));

		float min_ratio = xratio < yratio ? xratio : yratio;
		px_ += -px / zoom_ratio_ * 0.5f;
		py_ += py / zoom_ratio_ * 0.5f;
		zoom_ratio_ *= min_ratio;

		glScalef(min_ratio, min_ratio, 1.0f);
		parent_->setWindowTitle(windowTitle + QString(" - zoom x") + QString(std::to_string(zoom_ratio_).c_str()));
	}

	void GLWidget::dezoom()
	{
		glLoadIdentity();
		zoom_ratio_ = 1.0f;
		px_ = 0.0f;
		py_ = 0.0f;
		parent_->setWindowTitle(windowTitle);
	}

	void GLWidget::swap_selection_corners(holovibes::Rectangle& selection)
	{
		const int x_top_left = selection.topLeft().x();
		const int y_top_left = selection.topLeft().y();
		const int x_bottom_right = selection.bottomRight().x();
		const int y_bottom_rigth = selection.bottomRight().y();

		QPoint tmp;

		if (x_top_left < x_bottom_right)
		{
			if (y_top_left > y_bottom_rigth)
			{
				//selection.horizontal_symetry();
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
				//selection.vertical_symetry();
			}
			else
			{
				// Vertical and horizontal swaps
				//selection.vertical_symetry();
				//selection.horizontal_symetry();
			}
		}
	}

	void GLWidget::bounds_check(holovibes::Rectangle& selection)
	{
		if (selection.bottomRight().x() < 0)
			selection.bottomRight().setX(0);
		if (selection.bottomRight().x() > frame_desc_.width)
			selection.bottomRight().setX(frame_desc_.width);

		if (selection.bottomRight().y() < 0)
			selection.bottomRight().setY(0);
		if (selection.bottomRight().y() > frame_desc_.height)
			selection.bottomRight().setY(frame_desc_.height);

		//selection = Rectangle(selection.topLeft(), );
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