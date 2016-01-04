#pragma once

# include "gui_gl_widget.hh"

namespace gui
{
  class GLWidgetColor : public GLWidget
  {
  public:
    GLWidgetColor(
      holovibes::Holovibes& holovibes,
      holovibes::Queue& queue,
      const unsigned width,
      const unsigned height,
      QWidget* parent = 0)
      : GLWidget(holovibes, queue, width, height, parent)
    {
    }

    virtual ~GLWidgetColor()
    {
    }

  private:
    virtual void set_texture_format() override
    {
      glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB, // Classic RGBA pixels, with alpha component preset to 1 (opaque).
        frame_desc_.width,
        frame_desc_.height,
        0,
        GL_RGB,
        frame_desc_.depth == 1 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT,
        nullptr);
    }
  };
}