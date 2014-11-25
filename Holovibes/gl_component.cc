#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#include "gl_component.hh"

namespace holovibes
{
  GLComponent::GLComponent(
    HWND hwnd,
    const camera::FrameDescriptor& frame_desc,
    int width,
    int height)
    : hwnd_(hwnd)
    , hdc_(GetDC(hwnd))
    , hrc_(nullptr)
    , frame_desc_(frame_desc)
    , buffer_(0)
    , cuda_buffer_(nullptr)
  {
    PIXELFORMATDESCRIPTOR pfd = get_pfd();

    const int pixel_format = ChoosePixelFormat(hdc_, &pfd);
    if (!pixel_format)
      throw std::runtime_error("[OPENGL] unable to find a suitable pixel format");

    if (!SetPixelFormat(hdc_, pixel_format, &pfd))
      throw std::runtime_error("[OPENGL] can not set the pixel format");

    hrc_ = wglCreateContext(hdc_);

    if (!hrc_)
      throw std::runtime_error("[OPENGL] unable to create GL context");

    if (!wglMakeCurrent(hdc_, hrc_))
      throw std::runtime_error("[OPENGL] unable to make current GL context");

    glewInit();
    gl_enable(width, height);
  }

  GLComponent::~GLComponent()
  {
    gl_disable();
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(hrc_);
    ReleaseDC(hwnd_, hdc_);
  }

  inline PIXELFORMATDESCRIPTOR GLComponent::get_pfd()
  {
    return PIXELFORMATDESCRIPTOR
    {
      sizeof(PIXELFORMATDESCRIPTOR), // Size of this Pixel Format Descriptor
      1,                             // Version Number
      PFD_DRAW_TO_WINDOW |           // Format Must Support Window
      PFD_SUPPORT_OPENGL |           // Format Must Support OpenGL
      PFD_DOUBLEBUFFER,              // Must Support Double Buffering
      PFD_TYPE_RGBA,                 // Request An RGBA Format
      16,                            // Select Our Color Depth, 8 bits / channel
      0, 0, 0, 0, 0, 0,              // Color Bits Ignored
      0,                             // No Alpha Buffer
      0,                             // Shift Bit Ignored
      0,                             // No Accumulation Buffer
      0, 0, 0, 0,                    // Accumulation Bits Ignored
      24,                            // 32 bit Z-Buffer (Depth Buffer)
      0,                             // No Stencil Buffer
      0,                             // No Auxiliary Buffer
      PFD_MAIN_PLANE,                // Main Drawing Layer
      0,                             // Reserved
      0, 0, 0                        // Layer Masks Ignored
    };
  }

  /*! \brief OpenGL configuration.
  * \param width Width of GL viewport.
  * \param height Height of GL viewport.
  */
  void GLComponent::gl_enable(
    int width,
    int height)
  {
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

    glViewport(0, 0, width, height);
  }

  void GLComponent::gl_disable()
  {
    /* Unregister buffer for access by CUDA. */
    cudaGraphicsUnregisterResource(cuda_buffer_);
    /* Destroy buffer name. */
    glDeleteBuffers(1, &buffer_);
    glDisable(GL_TEXTURE_2D);
  }

  void GLComponent::gl_draw(
    const void* src)
  {
    /* Map the buffer for access by CUDA. */
    cudaGraphicsMapResources(1, &cuda_buffer_);
    size_t buffer_size;
    void* buffer_ptr;
    cudaGraphicsResourceGetMappedPointer(&buffer_ptr, &buffer_size, cuda_buffer_);
    /* CUDA memcpy of the frame to opengl buffer. */
    cudaMemcpy(buffer_ptr, src, buffer_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
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
    glTexCoord2d(0.0, 0.0); glVertex2d(-1.0, +1.0);
    glTexCoord2d(1.0, 0.0); glVertex2d(+1.0, +1.0);
    glTexCoord2d(1.0, 1.0); glVertex2d(+1.0, -1.0);
    glTexCoord2d(0.0, 1.0); glVertex2d(-1.0, -1.0);
    glEnd();

    SwapBuffers(hdc_);
    gl_error_checking();
  }

  void GLComponent::gl_error_checking()
  {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
      std::cerr << "[GL] " << gluErrorString(error) << std::endl;
  }
}