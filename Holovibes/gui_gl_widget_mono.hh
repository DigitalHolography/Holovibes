#pragma once

#include "gui_gl_widget.hh"

namespace gui
{
  class GLWidgetMono : public GLWidget
  {
  public:
    GLWidgetMono(
      holovibes::Holovibes& holovibes,
      holovibes::Queue& queue,
      const unsigned width,
      const unsigned height,
      QWidget* parent = 0)
      : GLWidget(holovibes, queue, width, height, parent)
    {
    }

    virtual ~GLWidgetMono()
    {
    }

  private:
    virtual void set_texture_format() override
    {
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
    }
  };
}