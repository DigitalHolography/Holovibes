#include "gui_gl_window.hh"

namespace gui
{
      GuiGLWindow::GuiGLWindow(QPoint& pos,
      unsigned int width,
      unsigned int height,
      holovibes::Queue& queue,
      QWidget* parent)
    : QMainWindow(parent),
    gl_widget_(nullptr)
  {
    ui.setupUi(this);
    this->move(pos);
    this->resize(QSize(width, height));
    this->show();
    gl_widget_ = new GLWidget(queue, width, height, this);
    gl_widget_->show();
  }
  
  GuiGLWindow::~GuiGLWindow()
  {
    delete gl_widget_;
  }

  void GuiGLWindow::resizeEvent(QResizeEvent* e)
  {
    if (gl_widget_)
      gl_widget_->resizeFromWindow(e->size().width(), e->size().width());
    resize(e->size().width(), e->size().width());
  }
}