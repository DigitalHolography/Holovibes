#ifndef GUI_GL_WINDOW_HH_
# define GUI_GL_WINDOW_HH_

# include <QMainWindow>
# include <QResizeEvent>
# include <QShortcut>
# include "ui_gl_window.h"
# include "gui_gl_widget.hh"
# include "holovibes.hh"

namespace gui
{
  class GuiGLWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    GuiGLWindow(QPoint& pos,
      unsigned int width,
      unsigned int height,
      holovibes::Holovibes& h,
      holovibes::Queue& q,
      QWidget* parent = 0);
    ~GuiGLWindow();

    void resizeEvent(QResizeEvent* e) override;

    GLWidget& get_gl_widget() const
    {
      return *gl_widget_;
    }

  public slots:
    void full_screen();
    void maximized_screen();
    void default_screen();

  private:
    Ui::GLWindow ui;
    GLWidget* gl_widget_;

    QShortcut* full_screen_;
    QShortcut* maximized_screen_;
    QShortcut* default_screen_;
  };
}

#endif /* !GUI_GL_WINDOW_HH_ */