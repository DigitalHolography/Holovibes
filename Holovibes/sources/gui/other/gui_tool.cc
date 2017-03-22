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

	/*void GuiTool::display_error(const std::string msg)
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
	*/
}