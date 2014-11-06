#ifndef GUI_GL_WINDOW_HH_
# define GUI_GL_WINDOW_HH_

# include <QMainWindow>
# include "ui_gl_window.h"

namespace gui
{
  class GuiGLWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    GuiGLWindow(QWidget *parent = 0);
    ~GuiGLWindow();

  private:
    Ui::GLWindow ui;
  };
}

#endif /* !GUI_GL_WINDOW_HH_ */