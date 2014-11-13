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
    GuiGLWindow(QWidget *parent = 0);
    ~GuiGLWindow();

    void resizeEvent(QResizeEvent* e) override;

  private:
    Ui::GLWindow ui;
  };
}

#endif /* !GUI_GL_WINDOW_HH_ */