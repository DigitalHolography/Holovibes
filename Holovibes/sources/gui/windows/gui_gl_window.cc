/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "gui_gl_window.hh"
#include "holovibes.hh"

#define DEFAULT_GLWIDGET_SIZE 600

namespace gui
{
	GuiGLWindow::GuiGLWindow(const QPoint& pos,
		const unsigned int width,
		const unsigned int height,
		float rotation,
		holovibes::Holovibes& h,
		holovibes::Queue& q,
		window_kind wk,
		QWidget* parent)
		: QMainWindow(parent),
		gl_widget_(nullptr),
		full_screen_(nullptr),
		maximized_screen_(nullptr)
	{
		if (wk == window_kind::DIRECT)
		{
			Ui::GLWindow ui;
			ui.setupUi(this);

			setWindowIcon(QIcon("icon1.ico"));

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
			move(pos);
			resize(QSize(width, height));
			show();

			// Default displaying format is 16-bits, monochrome.
			gl_widget_.reset(new GLWidget(h, q, width, height, this));
			gl_widget_->show();
		}
		else if (wk == window_kind::SLICE_VIEW)
		{
			Ui::GLSliceWindow ui;
			ui.setupUi(this);
			move(pos);
			resize(QSize(width, height));
			show();
			widget.reset(new SliceWidget(q, width, height, rotation, this));
			widget->show();
		}
	}

	GuiGLWindow::~GuiGLWindow()
	{
		delete full_screen_;
		delete maximized_screen_;
		delete default_screen_;
		gl_widget_.reset(nullptr);
		widget.reset(nullptr);
	}

	void GuiGLWindow::resizeEvent(QResizeEvent* e)
	{
		const QSize	s = e->size();
		const uint	min_dim = std::min(s.width(), s.height());

		if (gl_widget_)
		{
			
			// TODO: remove the dynamic cast when the widget_slice has been cleared up & its gonna SEGFAULT
			auto *ptr = dynamic_cast<GLWidget*>(gl_widget_.get());
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
				gl_widget_->move(s.width() / 2 - gl_widget_->width() / 2, 0);
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