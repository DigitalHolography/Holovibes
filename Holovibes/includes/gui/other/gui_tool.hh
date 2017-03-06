/*! \file
*
* Main class of the GUI. It regroup most of the Qt slots used for user actions. */
#pragma once

# include <QMessageBox>
# include <string>

# include "holovibes.hh"

namespace gui
{
	class MainWindow;
}

namespace gui
{

	class GuiTool
	{

	public:
		//TODO:
		GuiTool(holovibes::Holovibes& holovibes, MainWindow* mainWindow);

		~GuiTool();

		/*! \brief Display error message
		** \param msg error message
		*/
		//void display_error(std::string msg);
		/*! \brief Display information message
		** \param msg information message
		*/
		//void display_info(std::string msg);

		/*! \brief Check if direct button is enabled  */
		//bool is_direct_mode();

		/*! \brief holovibes_ getter */
		//holovibes::Holovibes& get_holovibes();

		//QObject* findChild(QString name);

	private:

		/*! Reference to Holovibes object */
		holovibes::Holovibes& holovibes_;

		/*! Pointer to MainWindowPanel */
		MainWindow* main_window_;

	};
}
