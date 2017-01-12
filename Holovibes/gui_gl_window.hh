/*! \file
 *
 * QMainWindow overload used to display the real time OpenGL frame  */
#pragma once

# include <QMainWindow>
# include <QResizeEvent>
# include <QShortcut>
# include <memory>

# include "ui_gl_window.h"
# include "gui_gl_widget.hh"

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
			QWidget* parent = nullptr);

		/* \brief GuiGLWindow destructor */
		~GuiGLWindow();

		/*! \brief Call when windows is resize */
		void resizeEvent(QResizeEvent* e) override;

		/*! \brief Returns a reference to a GLWidget object. */
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
		std::unique_ptr<GLWidget> gl_widget_;

		/*! \{ \name Screen modes keyboard shortcuts */
		QShortcut* full_screen_;
		QShortcut* maximized_screen_;
		QShortcut* default_screen_;
		/*! \} */
	};
}