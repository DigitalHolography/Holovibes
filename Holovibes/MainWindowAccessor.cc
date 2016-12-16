#include "MainWindowAccessor.hh"

namespace gui
{
	void MainWindowAccessor::setMainWindow(MainWindow *ref)
	{
		GetInstance().stored_ = ref;
	}
	MainWindow *MainWindowAccessor::getMainWindow(void)
	{
		return stored_;
	}
}