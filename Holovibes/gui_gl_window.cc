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
}