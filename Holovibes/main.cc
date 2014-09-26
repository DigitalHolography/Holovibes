#include "stdafx.h"
#include <iostream>
#include "camera.hh"
#include "xiq_camera.hh"
#include "gl_window.hh"

using namespace camera;
using namespace gui;

#define WIDTH 512
#define HEIGHT 512

int main()
{
  Camera* c = new XiqCamera();

  c->init_camera();

  GLWindow w;
  w.wnd_class_register();
  w.wnd_init("Test", WIDTH, HEIGHT);
  w.gl_init();
  w.wnd_show();

  // OpenGL
  GLuint texture;
  glGenTextures(1, &texture);
  glViewport(0, 0, WIDTH, HEIGHT);

  glEnable(GL_TEXTURE_2D);
  glEnable(GL_QUADS);

  c->start_acquisition();

  while (true)
  {  
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    void* frame = c->get_frame();

    glTexImage2D(
      GL_TEXTURE_2D,
      0,
      GL_LUMINANCE,
      2048,
      2048,
      0,
      GL_LUMINANCE,
      GL_UNSIGNED_BYTE,
      frame);

    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0); glVertex2d(-1.0, +1.0);
    glTexCoord2d(1.0, 0.0); glVertex2d(+1.0, +1.0);
    glTexCoord2d(1.0, 1.0); glVertex2d(+1.0, -1.0);
    glTexCoord2d(0.0, 1.0); glVertex2d(-1.0, -1.0);
    glEnd();

    SwapBuffers(w.get_hdc());
    glDeleteTextures(1, &texture);
  }

  c->stop_acquisition();

  glDisable(GL_QUADS);
  glDisable(GL_TEXTURE_2D);

  w.gl_free();

  c->shutdown_camera();

  return 0;
}