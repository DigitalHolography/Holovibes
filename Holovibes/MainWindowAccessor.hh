#pragma once

# include "main_window.hh"

namespace gui
{
	class MainWindow;
}

namespace gui
{
	/*! \class MainWindowAccessor
	** Singleton Class that containt a pointer to MainWindow
	**
	** Used to refresh Holovibes UI when MainWindow can't be accessed such as
	** try and catch in the refresh functions in pipe class.
	**
	** Pointer to MainWindow is get at Holovibes startup
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