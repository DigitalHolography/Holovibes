#include "gui_gl_widget.hh"

namespace gui
{
  GLWidget::GLWidget(holovibes::Queue& q, unsigned int width, unsigned int height, QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
    q_(q),
    width_(width),
    height_(height),
    fd_(q_.get_frame_desc()),
    texture_(0)
  {
    frame_ = malloc(fd_.width * fd_.height * fd_.depth);
  }

  GLWidget::~GLWidget()
  {
    free(frame_);
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
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_QUADS);

    glGenTextures(1, &texture_);
    glViewport(0, 0, width_, height_);
  }

  void GLWidget::resizeGL(int width, int height)
  {
    glViewport(0, 0, width, height);
  }

  void GLWidget::paintGL()
  {
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (fd_.endianness == camera::BIG_ENDIAN)
      glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_TRUE);
    else
      glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);


    unsigned int frame_size = fd_.width * fd_.height * fd_.depth;
    cudaMemcpy(frame_, q_.get_last_images(1), frame_size, cudaMemcpyDeviceToHost);

    glTexImage2D(
      GL_TEXTURE_2D,
      /* Base image level. */
      0,
      GL_LUMINANCE,
      fd_.width,
      fd_.height,
      /* border: This value must be 0. */
      0,
      GL_LUMINANCE,
      /* Unsigned byte = 1 byte, Unsigned short = 2 bytes. */
      fd_.depth == 1 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT,
      /* Pointer to image data in memory. */
      frame_);

    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0); glVertex2d(-1.0, +1.0);
    glTexCoord2d(1.0, 0.0); glVertex2d(+1.0, +1.0);
    glTexCoord2d(1.0, 1.0); glVertex2d(+1.0, -1.0);
    glTexCoord2d(0.0, 1.0); glVertex2d(-1.0, -1.0);
    glEnd();

    update();
  }

  void GLWidget::resizeFromWindow(int width, int height)
  {
    resizeGL(width, height);
    resize(QSize(width, height));
  }
}