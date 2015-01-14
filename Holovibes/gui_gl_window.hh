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
  /*! \class GuiGLWindow 
  **
  ** QMainWindow overload used to display the real time OpenGL frame.
  */
  class GuiGLWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    /*! \brief GuiGLWindow constructor 
    **
    ** \param pos initial position of the window
    ** \param width width of the window in pixels
    ** \param height height of the window in pixels
    ** \param h holovibes object
    ** \param q Queue from where to grab frames to display
    ** \param parent Qt parent
    */
    GuiGLWindow(QPoint& pos,
      unsigned int width,
      unsigned int height,
      holovibes::Holovibes& h,
      holovibes::Queue& q,
      QWidget* parent = 0);
    /* \brief GuiGLWindow destructor */
    ~GuiGLWindow();

    void resizeEvent(QResizeEvent* e) override;

    GLWidget& get_gl_widget() const
    {
      return *gl_widget_;
    }

  public slots:
    /*! \brief Set window to fullscreen mode */
    void full_screen();
    /*! \brief Set window to the maximum dimension of the screen */
    void maximized_screen();
    /*! \brief Set window back to normal default mode */
    void default_screen();

  private:
    Ui::GLWindow ui;
    /*! GL widget, it updates itself */
    GLWidget* gl_widget_;

    /*! \{ \name Screen modes keyboard shortcuts */
    QShortcut* full_screen_;
    QShortcut* maximized_screen_;
    QShortcut* default_screen_;
    /*! \} */
  };
}

#endif /* !GUI_GL_WINDOW_HH_ */