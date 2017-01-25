/*! \file
 *
 * QMainWindow overload used to display the real time OpenGL frame  */
#pragma once

# include <QMainWindow>
# include <QResizeEvent>
# include <QShortcut>
# include <memory>

# include "ui_gl_window.h"
# include "ui_gl_slice_window.h"
# include "gui_gl_widget.hh"
# include "gui_gl_widget_slice.hh"
# include "slice_widget.hh"

/* Forward declarations. */
namespace holovibes
{
	class Holovibes;
	class Queue;
}

namespace gui
{
	/*! \brief QMainWindow overload used to display the real time OpenGL frame. */
	class GuiGLWindow : public QMainWindow
	{
		Q_OBJECT

	public:

		enum window_kind
		{
			DIRECT,
			SLICE_XZ
		};

		/*! \brief GuiGLWindow constructor
		**
		** \param pos initial position of the window
		** \param width width of the window in pixels
		** \param height height of the window in pixels
		** \param h holovibes object
		** \param q Queue from where to grab frames to display
		** \param parent Qt parent
		*/
		GuiGLWindow(const QPoint& pos,
			const unsigned int width,
			const unsigned int height,
			holovibes::Holovibes& h,
			holovibes::Queue& q,
		    window_kind wk = window_kind::DIRECT,
			QWidget* parent = nullptr);

		/* \brief GuiGLWindow destructor */
		~GuiGLWindow();

		/*! \brief Call when windows is resize */
		void resizeEvent(QResizeEvent* e) override;

		/*! \brief Returns a reference to a GLWidget object. */
		GLWidget& get_gl_widget() const
		{
			//TODO: : remove the dynamic cast when the widget_slice has been cleared up & its gonna SEGFAULT
			return *(dynamic_cast<GLWidget*>(gl_widget_.get()));
		}

		public slots:
		/*! \brief Set window to fullscreen mode */
		void full_screen();
		/*! \brief Set window to the maximum dimension of the screen */
		void maximized_screen();
		/*! \brief Set window back to normal default mode */
		void default_screen();

	private:
		/*! GL widget, it updates itself */
		std::unique_ptr<QGLWidget>		gl_widget_;
		std::unique_ptr<SliceWidget>	widget;

		/*! \{ \name Screen modes keyboard shortcuts */
		QShortcut* full_screen_;
		QShortcut* maximized_screen_;
		QShortcut* default_screen_;
		/*! \} */
	};
}