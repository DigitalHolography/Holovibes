#include <QOpenGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

#include "gui_gl_widget.hh"

namespace gui
{
  GLWidget::GLWidget(
    holovibes::Queue& q,
    unsigned int width,
    unsigned int height,
    QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
    , QOpenGLFunctions()
    , timer_(this)
    , width_(width)
    , height_(height)
    , queue_(q)
    , frame_desc_(q.get_frame_desc())
    , buffer_(0)
    , cuda_buffer_(nullptr)
    , is_selection_enabled_(false)
    , px_(0.0f)
    , py_(0.0f)
  {
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
      float selection_color[4] = { 0.0f, 0.3f, 0.0f, 0.4f };
      selection_rect(startx_, starty_, endx_, endy_, selection_color);
    }

    gl_error_checking();
  }

  void GLWidget::mousePressEvent(QMouseEvent* e)
  {
    is_selection_enabled_ = true;
    startx_ = (e->x() * frame_desc_.width) / width() + px_;
    starty_ = (e->y() * frame_desc_.height) / height() + py_;
  }

  void GLWidget::mouseMoveEvent(QMouseEvent* e)
  {
    endx_ = (e->x() * frame_desc_.width) / width() + px_;
    endy_ = (e->y() * frame_desc_.height) / height() + py_;
  }

  void GLWidget::mouseReleaseEvent(QMouseEvent* e)
  {
    endx_ = (e->x() * frame_desc_.width) / width() + px_;
    endy_ = (e->y() * frame_desc_.height) / height() + py_;
    is_selection_enabled_ = false;

    std::cout << "start (" << startx_ << ", " << starty_ << ")" << std::endl;
    std::cout << "end (" << endx_ << ", " << endy_ << ")" << std::endl;
    std::cout << "mouse release eve(px, py): (" << px_ << ", " << py_ << ")" << std::endl;
    //FIXME
    zoom();
  }

  void GLWidget::selection_rect(int startx, int starty, int endx, int endy, float color[4])
  {
    float xmax = frame_desc_.width;
    float ymax = frame_desc_.height;
    float nstartx = (2.0f * (float)startx) / xmax - 1.0f;
    float nstarty = -1.0f * ((2.0f * (float)starty) / ymax - 1.0f);
    float nendx = (2.0f * (float)endx) / xmax - 1.0f;
    float nendy = -1.0f * ((2.0f * (float)endy) / ymax - 1.0f);

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

  void GLWidget::zoom()
  {
    int selection_width = endx_ - startx_;
    int selection_height = endy_ - starty_;

    float xratio = (float)frame_desc_.width / (float)selection_width;
    float yratio = (float)frame_desc_.height / (float)selection_height;

    float min_ratio = xratio < yratio ? xratio : yratio;

    // Zoom
    glScalef(min_ratio, min_ratio, 1.0f);

    int originx = 0 + px_;
    int originy = 0 + py_;

    // Sign of the translation
    int px_sign = originx < startx_ ? -1 : 1;
    int py_sign = originy < starty_ ? -1 : 1;

    // Projections on x and y axis / translation vector coordinates
    // Centering the selection zone by adding selection_dim / 2
    int px = px_sign * abs(startx_ + selection_width / 2 - originx);
    int py = py_sign * abs(starty_ + selection_height / 2 - originy);

    px -= px_;
    py -= py_;

    std::cout << "translate(px, py): (" << px << ", " << py << ")" << std::endl;

    // Normalization for OpenGL
    float npx = (2.0f * (float)px) / frame_desc_.width;
    float npy = -1.0f * (2.0f * (float)py) / frame_desc_.height;

    // Translation in the middle of the window
    glTranslatef(npx + 1.0f, npy - 1.0f, 1.0f);

    // Keeping previouses translations
    px_ -= px + frame_desc_.width / 2;
    py_ -= py + frame_desc_.height / 2;
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
}