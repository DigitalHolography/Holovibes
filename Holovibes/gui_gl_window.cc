#include "stdafx.h"
#include "gui_gl_window.hh"

namespace holovibes
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