#pragma once

# include "main_window.hh"

namespace gui
{
	class MainWindow;
}

namespace gui
{
	/*! \class MainWindowAccessor
	** Singleton Class that contains a pointer to MainWindow
	**
	** Used to refresh Holovibes UI when MainWindow can't be accessed directly such as
	** try and catch in refresh functions of pipe class.
	**
	** Pointer to MainWindow is get at Holovibes startup
	**
	** THIS CLASS SHOULD BE USED AS LEAST AS POSSIBLE
	*/
	class MainWindowAccessor
	{
	public:
		static MainWindowAccessor& GetInstance()
		{
			static MainWindowAccessor win;
			return win;
		}
		void setMainWindow(MainWindow *ref);
		MainWindow *getMainWindow(void);

	private:
		MainWindowAccessor() {}
		MainWindowAccessor(const MainWindowAccessor&);
		MainWindowAccessor& operator=(const MainWindowAccessor&);
		MainWindow *stored_;
	};
}