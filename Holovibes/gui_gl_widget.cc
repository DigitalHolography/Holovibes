#include <QOpenGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

#include "gui_gl_widget.hh"

namespace gui
{
  GLWidget::GLWidget(
    holovibes::Holovibes& h,
    holovibes::Queue& q,
    unsigned int width,
    unsigned int height,
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
    , is_zoom_enabled_(true)
    , is_average_enabled_(false)
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
  }

  GLWidget::~GLWidget()
  {
    /* Unregister buffer for access by CUDA. */
    cudaGraphicsUnregisterResource(cuda_buffer_);
    /* Destroy buffer name. */
    glDeleteBuffers(1, &buffer_);
    glDisable(GL_TEXTURE_2D);
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
    /* Creates and initialize a buffer object's data store. */
    glBufferData(
      GL_TEXTURE_BUFFER,
      frame_desc_.frame_size(),
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
    cudaGraphicsMapResources(1, &cuda_buffer_);
    size_t buffer_size;
    void* buffer_ptr;
    cudaGraphicsResourceGetMappedPointer(&buffer_ptr, &buffer_size, cuda_buffer_);
    /* CUDA memcpy of the frame to opengl buffer. */
    cudaMemcpy(buffer_ptr, frame, buffer_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    /* Unmap the buffer for access by CUDA. */
    cudaGraphicsUnmapResources(1, &cuda_buffer_);

    /* Bind the buffer object to the target GL_PIXEL_UNPACK_BUFFER.
     * This affects glTexImage2D command. */
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_);

    if (frame_desc_.endianness == camera::BIG_ENDIAN)
      glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_TRUE);
    else
      glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(
      GL_TEXTURE_2D,
      /* Base image level. */
      0,
      GL_LUMINANCE,
      frame_desc_.width,
      frame_desc_.height,
      /* border: This value must be 0. */
      0,
      GL_LUMINANCE,
      /* Unsigned byte = 1 byte, Unsigned short = 2 bytes. */
      frame_desc_.depth == 1 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT,
      /* Pointer to image data in memory. */
      NULL);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBegin(GL_QUADS);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glTexCoord2d(0.0, 0.0); glVertex2d(-1.0, +1.0);
    glTexCoord2d(1.0, 0.0); glVertex2d(+1.0, +1.0);
    glTexCoord2d(1.0, 1.0); glVertex2d(+1.0, -1.0);
    glTexCoord2d(0.0, 1.0); glVertex2d(-1.0, -1.0);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    if (is_selection_enabled_)
    {
      if (is_average_enabled_)
      {
        float signal_color[4] = { 1.0f, 0.0f, 0.5f, 0.4f };
        selection_rect(signal_selection_, signal_color);

        float noise_color[4] = { 0.26f, 0.56f, 0.64f, 0.4f };
        selection_rect(noise_selection_, noise_color);
      }
      else // if (is_zoom_enabled_)
      {
        float selection_color[4] = { 0.0f, 0.5f, 0.0f, 0.4f };
        selection_rect(selection_, selection_color);
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
          ( e->y() * frame_desc_.height) / height());
      }
      else
        if (is_zoom_enabled_)
          dezoom();
  }

  void GLWidget::mouseMoveEvent(QMouseEvent* e)
  {
    if (is_selection_enabled_)
    {
      selection_.bottom_right = holovibes::Point2D(
        (e->x() * frame_desc_.width) / width(),
         (e->y() * frame_desc_.height) / height());

      if (is_average_enabled_)
      {
        if (is_signal_selection_)
          signal_selection_ = selection_;
        else // Noise selection
          noise_selection_ = selection_;
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

      selection_.bottom_left = holovibes::Point2D(
        selection_.top_left.x,
         (e->y() * frame_desc_.height) / height());

      selection_.top_right = holovibes::Point2D(
        (e->x() * frame_desc_.width) / width(),
        selection_.top_left.y);

      bounds_check(selection_);
      swap_selection_corners(selection_);

      if (is_average_enabled_)
      {
        if (is_signal_selection_)
        {
          signal_selection_ = selection_;
          h_.get_compute_desc().signal_zone = signal_selection_;
        }
        else // Noise selection
        {
          noise_selection_ = selection_;
          h_.get_compute_desc().noise_zone = noise_selection_;
        }

        is_signal_selection_ = !is_signal_selection_;
        selection_ = holovibes::Rectangle();
      }
      else // if (is_zoom_enabled_)
      {
        is_selection_enabled_ = false;

        if (selection_.top_left != selection_.bottom_right)
          zoom(selection_);

        selection_ = holovibes::Rectangle();
      }
    }
  }

  void GLWidget::selection_rect(const holovibes::Rectangle& selection, float color[4])
  {
    float xmax = frame_desc_.width;
    float ymax = frame_desc_.height;

    float nstartx = (2.0f * (float)selection.top_left.x) / xmax - 1.0f;
    float nstarty = -1.0f * ((2.0f * (float)selection.top_left.y) / ymax - 1.0f);
    float nendx = (2.0f * (float)selection.bottom_right.x) / xmax - 1.0f;
    float nendy = -1.0f * ((2.0f * (float)selection.bottom_right.y) / ymax - 1.0f);

    nstartx -= px_;
    nstarty -= py_;
    nendx -= px_;
    nendy -= py_;

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

  void GLWidget::zoom(const holovibes::Rectangle& selection)
  {
    // Translation
    // Destination point is center of the window (OpenGL coords)
    float xdest = 0.0f;
    float ydest = 0.0f;

    // Source point is center of the selection zone (normal coords)
    int xsource = selection.top_left.x + ((selection.bottom_right.x - selection.top_left.x) / 2);
    int ysource = selection.top_left.y + ((selection.bottom_right.y - selection.top_left.y) / 2);

    // Normalizing source points to OpenGL coords
    float nxsource = (2.0f * (float)xsource) / (float)frame_desc_.width - 1.0f;
    float nysource = -1.0f * ((2.0f * (float)ysource) / (float)frame_desc_.height - 1.0f);

    // Projection of the translation
    float px = xdest - nxsource;
    float py = ydest - nysource;

    // Zoom ratio
    float xratio = (float)frame_desc_.width / ((float)selection.bottom_right.x - (float)selection.top_left.x);
    float yratio = (float)frame_desc_.height / ((float)selection.bottom_right.y - (float)selection.top_left.y);

    float min_ratio = xratio < yratio ? xratio : yratio;
    zoom_ratio_ *= min_ratio;

    // Translation
    glTranslatef(px * min_ratio, py * min_ratio, 0.0f);

    // Rescale
    glScalef(min_ratio, min_ratio, 1.0f);

    px_ += px * zoom_ratio_;
    py_ += py * zoom_ratio_;

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
    int x_top_left = selection.top_left.x;
    int y_top_left = selection.top_left.y;
    int x_bottom_right = selection.bottom_right.x;
    int y_bottom_rigth = selection.bottom_right.y;

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
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
      std::cerr << "[GL] " << glGetString(error) << std::endl;
  }

  void GLWidget::resizeFromWindow(int width, int height)
  {
    resizeGL(width, height);
    resize(QSize(width, height));
  }

  void GLWidget::set_average_mode(bool value)
  {
    is_average_enabled_ = value;
    is_zoom_enabled_ = !value;
  }
}