#include "gui_tool.hh"
#include "main_window.hh"

namespace gui
{

	GuiTool::GuiTool(holovibes::Holovibes& holovibes, MainWindow* mainWindow)
		: holovibes_(holovibes)
		, main_window_(mainWindow)
	{}

	GuiTool::~GuiTool()
	{}

	void GuiTool::display_error(const std::string msg)
	{
		QMessageBox msg_box;
		msg_box.setText(QString::fromLatin1(msg.c_str()));
		msg_box.setIcon(QMessageBox::Critical);
		msg_box.exec();
	}

	void GuiTool::display_info(const std::string msg)
	{
		QMessageBox msg_box;
		msg_box.setText(QString::fromLatin1(msg.c_str()));
		msg_box.setIcon(QMessageBox::Information);
		msg_box.exec();
	}

	bool GuiTool::is_direct_mode()
	{
		return holovibes_.get_compute_desc().compute_mode == holovibes::ComputeDescriptor::compute_mode::DIRECT;
	}

	holovibes::Holovibes& GuiTool::get_holovibes()
	{
		return holovibes_;
	}

	QObject* GuiTool::findChild(QString name)
	{
		return main_window_->findChild<QObject*>(name);
	}

}