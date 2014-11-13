#include "stdafx.h"
#include "gui_gl_window.hh"

namespace gui
{
  GuiGLWindow::GuiGLWindow(QWidget *parent)
    : QMainWindow(parent)
  {
    ui.setupUi(this);
  }
  
  GuiGLWindow::~GuiGLWindow()
  {
  }

  void GuiGLWindow::resizeEvent(QResizeEvent* e)
  {
    GLWidget* gl_widget = this->findChild<GLWidget*>("GL");

    if (gl_widget)
      gl_widget->resizeFromWindow(e->size().width(), e->size().width());

    resize(e->size().width(), e->size().width());
  }
}