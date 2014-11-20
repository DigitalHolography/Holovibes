#ifndef GUI_GL_WINDOW_HH_
# define GUI_GL_WINDOW_HH_

# include <QMainWindow>
# include <QResizeEvent>
# include "ui_gl_window.h"
# include "gui_gl_widget.hh"

namespace gui
{
  class GuiGLWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    GuiGLWindow(QPoint& pos,
      unsigned int width,
      unsigned int height,
      holovibes::Queue& queue,
      QWidget* parent = 0);
    ~GuiGLWindow();

    void resizeEvent(QResizeEvent* e) override;

  private:
    Ui::GLWindow ui;
    GLWidget* gl_widget_;
  };
}

#endif /* !GUI_GL_WINDOW_HH_ */