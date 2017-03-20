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
