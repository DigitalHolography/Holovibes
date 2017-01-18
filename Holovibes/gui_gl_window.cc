#include "gui_gl_window.hh"
#include "holovibes.hh"

#define DEFAULT_GLWIDGET_SIZE 600

namespace gui
{
	GuiGLWindow::GuiGLWindow(const QPoint& pos,
		const unsigned int width,
		const unsigned int height,
		holovibes::Holovibes& h,
		holovibes::Queue& q,
		window_kind wk,
		QWidget* parent)
		: QMainWindow(parent)
		, gl_widget_(nullptr)
		, full_screen_(nullptr)
		, maximized_screen_(nullptr)
	{
		//TODO:
		if (wk == window_kind::DIRECT)
		{
			Ui::GLWindow ui;
			ui.setupUi(this);

			this->setWindowIcon(QIcon("icon1.ico"));

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

			// Default displaying format is 16-bits, monochrome.
			gl_widget_.reset(new GLWidget(h, q, width, height, this));
		}
		else
		{
			Ui::GLSliceWindow ui;
			ui.setupUi(this);
			this->move(pos);
			this->resize(QSize(width, height));
			this->show();
			gl_widget_.reset(new GLWidgetSlice(h, q, width, height, this));
		}
		gl_widget_->show();
	}

	GuiGLWindow::~GuiGLWindow()
	{
		gl_widget_.reset(nullptr);
	}

	void GuiGLWindow::resizeEvent(QResizeEvent* e)
	{
		unsigned int min_dim = e->size().width() < e->size().height() ? e->size().width() : e->size().height();

		if (gl_widget_)
		{
			
			// TODO: remove the dynamic cast when the widget_slice has been cleared up & its gonna SEGFAULT
			auto* ptr = dynamic_cast<GLWidget*>(gl_widget_.get());
			if (ptr)
				ptr->resizeFromWindow(min_dim, min_dim);
			else
			dynamic_cast<GLWidgetSlice*>(gl_widget_.get())->resizeFromWindow(min_dim, min_dim);
			
			if (windowState() != Qt::WindowFullScreen)
			{
				resize(min_dim, min_dim);
				gl_widget_->move(0, 0);
			}
			else
				gl_widget_->move(e->size().width() >> 1 - gl_widget_->width() >> 1, 0);
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