#include "gui_gl_window.hh"

#define DEFAULT_GLWIDGET_SIZE 600

namespace gui
{
      GuiGLWindow::GuiGLWindow(QPoint& pos,
      unsigned int width,
      unsigned int height,
      holovibes::Queue& queue,
      QWidget* parent)
    : QMainWindow(parent),
    gl_widget_(nullptr),
    full_screen_(nullptr),
    maximized_screen_(nullptr)
  {
    ui.setupUi(this);

    // Keyboard shortcuts
    full_screen_ = new QShortcut(QKeySequence("Ctrl+f"), this);
    full_screen_->setContext(Qt::ApplicationShortcut);
    connect(full_screen_, SIGNAL(activated()), this, SLOT(full_screen()));

    maximized_screen_ = new QShortcut(QKeySequence("Ctrl+m"), this);
    connect(maximized_screen_, SIGNAL(activated()), this, SLOT(maximized_screen()));
    maximized_screen_->setContext(Qt::ApplicationShortcut);

    default_screen_ = new QShortcut(QKeySequence("Esc"), this);
    connect(default_screen_, SIGNAL(activated()), this, SLOT(default_screen()));
    default_screen_->setContext(Qt::ApplicationShortcut);

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
    unsigned int min_dim;

    min_dim = e->size().width() < e->size().height() ? e->size().width() : e->size().height();

    if (gl_widget_)
    {
      gl_widget_->resizeFromWindow(min_dim, min_dim);

      if (windowState() != Qt::WindowFullScreen)
      {
        resize(min_dim, min_dim);
        gl_widget_->move(0, 0);
      }
      else
        gl_widget_->move(e->size().width() / 2 - gl_widget_->width() / 2, 0);
    }
  }

  void GuiGLWindow::full_screen()
  {
    setWindowState(windowState() ^ Qt::WindowFullScreen);
    show();
  }

  void GuiGLWindow::maximized_screen()
  {
    setWindowState(windowState() ^ Qt::WindowMaximized);
    show();
  }

  void GuiGLWindow::default_screen()
  {
    setWindowState(Qt::WindowNoState);
    show();
  }
}