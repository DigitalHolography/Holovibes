#pragma once

# include "main_window.hh"

namespace gui
{
	class MainWindow;
}

namespace gui
{
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