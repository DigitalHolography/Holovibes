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
* Singleton managing the 'Infos' GroupBox. TODO: recode this class. */
#pragma once

#include <mutex>
#include <thread>

#include <QObject>
#include <QProgressBar>
#include <QString>
#include <QTextEdit>
#include <QThread>

#include "gui_group_box.hh"
#include "frame_desc.hh"

namespace holovibes
{
	namespace gui
	{
	/*! \brief InfoManager is a singleton use to control info printed in infoLayout
	**
	** You can use it from anywere, using update/remove info to print what you want
	** or get  a progress_bar to inform your progression */
	class InfoManager : public QThread
	{
		Q_OBJECT
			signals :
		/*! Inform infoEdit_ about new text to display*/
		void update_text(const QString);
	public:

		enum InfoType // Order them the way you want them to be displayed
		{
			IMG_SOURCE,
			RECORDING,
			INPUT_SOURCE,
			INPUT_FPS,
			INPUT_QUEUE,
			OUTPUT_SOURCE,
			OUTPUT_QUEUE,
			RAW_OUTPUT_QUEUE,
			RENDERING_FPS,
			OUTPUT_THROUGHPUT,
			INPUT_THROUGHPUT,
			SAVING_THROUGHPUT,
			STFT_SLICE_CURSOR,
			STFT_ZONE,
			STFT_QUEUE,
			ERR,
			INFO
		};

		void			startDelError(const std::string& key);

		/*! Get the singleton, it's create on first call
		** \param ui must containt infoProgressBar and infoTextEdit in child*/
		static InfoManager *get_manager(gui::GroupBox *ui = nullptr);

		/*! Stop to refresh the info_panel display*/
		void stop_display();

		void insertInputSource(const camera::FrameDescriptor& fd);

		/*! Add your information until is remove, and call draw()
		** \param key is where you can access to your information
		** \param value is your information linked to key */
		void update_info(const std::string& key, const std::string& value);
		void remove_info(const std::string& key);
		void insert_info(const uint pos, const std::string& key, const std::string& value);
		void clear_infos();
		/*! Return progress_bar, you can use it as you want */
		QProgressBar *get_progress_bar();
	private:
		/*! Throwed if try instance InfoManager more once*/
		class ManagerNotInstantiate : std::exception
		{
			const char *what()
			{
				return "InfoManager is not instantiate, use InfoManager::get_manager with arg";
			}
		};
		enum class ThreadState {
			Null,
			Operating,
			Finish
		};

		/*! ctr */
		InfoManager(gui::GroupBox *ui);

		/*! Draw all current information */
		void draw();

		void run() override;

		void joinDelErrorThread();

		std::thread*	delError;
		ThreadState		flag;
		void		taskDelError(const std::string& key);

		/*! ui where find infoProgressBar and infoTextEdit */
		gui::GroupBox*  ui_;
		/*! infoProgressBar, you can get it with get_progress_bar() */
		QProgressBar*   progressBar_;
		/*! infoTextEdit, use to display informations*/
		QTextEdit*      infoEdit_;

		bool stop_requested_;

		/*! To prevent simultaneous access from multiple thread but one thread cant block himself 
		Private function don't need to lock the mutex because they are called from a public function */
		std::recursive_mutex mutex_;

		/*! Store all informations */
		std::vector<std::pair<std::string, std::string>>  infos_;
	};
}
}