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

#pragma once

# include <boost/algorithm/string.hpp>
# include <boost/filesystem.hpp>
# include <boost/property_tree/ptree.hpp>
# include <boost/property_tree/ini_parser.hpp>
# include <cstring>
# include <QDesktopServices>
# include <QFileDialog>
# include <QMainWindow>
# include <QMessageBox>
# include <QShortcut>
# include <sys/stat.h>
# include <thread>
# include <sstream>

// without namespace
# include "tools.hh"

// namespace camera
# include "camera_exception.hh"

// namespace gpib
# include "gpib_dll.hh"
# include "IVisaInterface.hh"
# include "gpib_controller.hh"
# include "gpib_exceptions.hh"

// namespace holovibes
# include "holovibes.hh"
# include "custom_exception.hh"
# include "info_manager.hh"
# include "options_descriptor.hh"

// namespace gui
# include "HoloWindow.hh"
# include "SliceWindow.hh"
# include "PlotWindow.hh"
# include "thread_csv_record.hh"
# include "thread_recorder.hh"
# include "ui_MainWindow.h"

#define GLOBAL_INI_PATH "holovibes.ini"

namespace holovibes
{
	namespace gui
	{
		/*! \class MainWindow
		**
		** Main class of the GUI. It regroup most of the Qt slots used for user actions.
		** These slots are divided into several sections:
		**
		** * Menu: every action in the menu (e-g: configuration of .ini, camera selection ...).
		** * Image rendering: phase number, p, z, lambda ...
		** * View: log scale, shifted corner, contrast ...
		** * Special: image ratio, average/ROI plot ...
		** * Record: record of raw frames, average/ROI file ...
		** * Import : making a file of raw data the image source
		** * Info : Various runtime informations on the program's state
		*/
		class MainWindow : public QMainWindow, public Observer
		{
			Q_OBJECT
			/* ---------- */
			#pragma region Public Methods
		public:
			/*! \brief Set keyboard shortcuts, set visibility and load default values from holovibes.ini.
			**
			** \param holovibes holovibes object
			** \param parent Qt parent (should be null as it is the GUI hierarchy top class)
			*/
			MainWindow(Holovibes& holovibes, QWidget *parent = 0);

			/*! \brief MainWindow destructor
			**
			** Destroy respectively compute and capture threads.
			*/
			~MainWindow();

			void notify() override;

			void notify_error(std::exception& e, const char* msg) override;
			#pragma endregion
			/* ---------- */
			#pragma region Public Slots
		public slots:
			/*! \brief Display information message
			** \param msg information message
			*/
			void display_message(QString message);
			/*! \{ \name Menu */
			/*! \brief Resize windows if one layout is toggled.
			**
			** b is unused
			*/
			void layout_toggled(bool b);
			/*! \brief Open holovibes configuration file */
			void configure_holovibes();
			/*! \brief Set camera to NONE
			**
			** Delete GL widget, destroy capture and/or compute thread then
			** set visibility to false.
			*/
			void camera_none();
			/*! \brief Change camera to Adimec */
			void camera_adimec();
			/*! \brief Change camera to EDGE */
			void camera_edge();
			/*! \brief Change camera to IDS */
			void camera_ids();
			/*! \brief Change camera to iXon */
			void camera_ixon();
			/*! \brief Change camera to Pike */
			void camera_pike();
			/*! \brief Change camera to Pixelfly */
			void camera_pixelfly();
			/*! \brief Change camera to XIQ */
			void camera_xiq();
			/*! \brief Open camera configuration file */
			void configure_camera();
			/*! \brief Display program's credits */
			void credits();
			/* \} */
			/*! \brief init the holovibes whenever changing rendering mode */
			void init_image_mode(QPoint& position, QSize& size);
			/*! \{ \name Image rendering
			**
			** Lots of these methods stick to the following scheme:
			**
			** * Get pipe
			** * Set visibility to false
			** * Check if value is correct/into slot specific bounds
			** * Update a value in FrameDescriptor of the holovibes object
			** * Request a pipe refresh
			** * Set visibility to true
			*/
			/*! \brief Set image mode either to direct or hologram mode
			**
			** Check if Camera has been enabled, then create a new GuiGLWindow keeping
			** its old position and size if it was previously opened, set visibility
			** and call notify().
			**
			** \param value true for direct mode, false for hologram mode.
			*/
			void set_image_mode();
			/*! \brief Called by set_image_mode if direct button is enabled  */
			void set_direct_mode();
			/*! \brief Called by set_image_mode if hologram button is clicked  */
			/* */
			void set_holographic_mode();
			/*! \brief Called by set_image_mode if complex algorithm is clicked  */
			/* */
			void refreshViewMode();
			/*! \brief Called by set_image_mode if Convolution button is clicked  */
			/* */
			void set_convolution_mode(const bool enable);
			/*! \brief Called by set_image_mode if Flowgraphy button is clicked  */
			/* */
			void set_flowgraphy_mode(const bool enable);

			/*! \brief Check if direct button is enabled  */
			bool is_direct_mode();
			/*! \brief Reset the GPU ressources and camera's record */
			void reset();
			/* launch the take reference mode that record one or several images and substract the mean of it to the others */
			void take_reference();
			/* launch the take slinding reference mode that record the last images and  substract the mean of it to the others */
			void take_sliding_ref();
			/* cancel the reference_taking mode */
			void cancel_take_reference();
			/* launch the filter2D mode that works in two steps. First, you need to select a rectangle on the displayed area and then
			** the computation can occurs.
			*/
			void set_filter2D();
			/* cancel the filter_2D mode */
			void cancel_filter2D();
			/*! \brief Set phase number (also called 'n' in papers)
			** \param value new phase number
			*/
			void setPhase();
			//void set_phase_number(int value);
			/*! \brief Set special buffer size
			** \param value new buffer size
			*/
			void set_special_buffer_size(int value);
			/*! \brief Set p-th frame to be displayed in OpenGl window
			** \param value new p
			*/
			void set_p_accu();
			void set_p(int value);
			/*! \brief Increment flography level */
			void set_flowgraphy_level(const int value);
			/*! \brief Increment p (useful for keyboard shortcuts) */
			void increment_p();
			/*! \brief Decrement p (useful for keyboard shortcuts) */
			void decrement_p();
			/*! \brief Set wavelength/lambda
			** \param value wavelength/lambda value in meters
			*/
			void set_wavelength(double value);
			/*! \brief Set z/distance to object
			** \param value z in meters
			*/
			void set_z(double value);
			/*! \brief Increment z (useful for keyboard shortcuts) */
			void increment_z();
			/*! \brief Decrement z (useful for keyboard shortcuts) */
			void decrement_z();
			/*! \brief Set z step (useful for keyboard shortcuts)
			** \param new z step
			*/
			void set_z_step(double value);
			/*! \brief Set algorithm
			** \param value algorithm "1FFT" or "2FFT"
			*/
			void set_algorithm(QString value);

			/*! \brief Set algorithm
			** \param value to set stft on/off
			*/
			void set_stft(bool b);

			/*! \brief create winndows
			** \and set the slices
			** \param value to set stft view on/off
			*/
			void stft_view(bool checked);
			void cancel_stft_slice_view();

			/*! \} */
			// \brief update how often the STFT is computed while STFT mode is activated
			void update_stft_steps(int value);

			/*! \{ \name View */
			/*! \brief Set view mode
			** \param value view mode: "magnitude", "squared magnitude", "argument",
			** "unwrapped argument", or "unwrapped argument 2".
			*/
			void set_view_mode(QString value);
			/*! Set the size of the unwrapping history window.
			*/
			void set_unwrap_history_size(int value);
			/*! Activate / Deactivate phase unwrapping 1d.
			*/
			void set_unwrapping_1d(const bool value);
			/*! Activate / Deactivate phase unwrapping 2d.
			*/
			void set_unwrapping_2d(const bool value);

			/*! \brief Set autofocus mode on
			**
			** Set GLWidget selection mode to AUTOFOCUS.
			** Check and set values mandatory for autofocus computation then connect end of
			** selection signal of OpenGl widget to request_autofocus() slot. Then whenever the
			** user has finished its selection, the request will be called.
			*/
			void set_accumulation(bool value);
			/*! \brief Set accmulation on or off
			*/
			void set_accumulation_level(int value);
			/*! \brief Set the number of image accmulated
			*/
			void set_z_min(const double value);
			/*! \brief Set z_min for autofocus
			*/
			void set_z_max(const double value);
			/*! \brief Set z_max for autofocus
			*/
			void set_z_iter(const int value);
			/*! \brief Set z_iter for autofocus
			*/
			void set_z_div(const int value);
			/*! \brief Set z_div for autofocus
			*/
			void set_autofocus_mode();
			/*! \brief Request to stop the autofocus currently
			**  occuring.
				*/
			void request_autofocus_stop();
			/*! \brief Enable or disable contrast mode
			** \param value true to enable coontrast, false otherwise.
			*/
			void set_import_pixel_size(const double value);
			/*! \brief Set import_pixel_size for autofocus
			*/
			void set_contrast_mode(bool value);
			/*! \brief Requet autocontrast action in pipe
			**					** It will automatically fill contrast minimum and maximum values.
			*/

			void pipe_refresh();
			// Safe pipe refresh

			void set_auto_contrast_cuts();
			void set_auto_contrast();
			/*! \brief Set contrast minimum value
			** \param value new contrast minimum value
			*/
			void set_contrast_min(double value);
			/*! \brief Set contrast maximum value
			** \param value new contrast maximum value
			*/
			void set_contrast_max(double value);
			/*! \brief Enable or disable logarithmic scale */
			void set_log_scale(bool value);
			/*! \brief Enable or diable shift corners algorithm */
			void set_shifted_corners(bool value);
			/*! \} */
			/*! \{ \name Special */
			/*! \brief Enable or disable vibrometry/image ratio mode */
			void set_vibro_mode(bool value);
			/*! \brief Set p-th frame for vibrometry mode
			** \param value new p vibrometry
			*/
			void set_p_vibro(int value);
			/*! \brief Set q-th frame for vibrometry mode
			** \param value new q vibrometry
			*/
			void set_q_vibro(int value);
			/*! \brief Enable or disable average/ROI mode */
			void set_average_mode(bool value);

			void activeSignalZone();
			void activeNoiseZone();

			/*! \brief Plot average/ROI computations */
			void set_average_graphic();
			/*! \brief Dispose average/ROI computations */
			void dispose_average_graphic();

			/*! \brief Browse average/ROI zone file for load/save */
			void browse_roi_file();
			/*! \brief Save ROI zone to file */
			void save_roi();
			/*! \brief Load ROI zone from file */
			void load_roi();
			/*! \} */

			/*! \brief Browse Convolution Matrix file for load */
			void browse_convo_matrix_file();
			/*! \brief Load Convolution Matrix from file */
			void load_convo_matrix();

			/*! \{ \name Record */
			/*! \brief Browse image record output file */
			void browse_file();
			/*! \brief Launch image record
			**
			** A ThreadRecord is used for image recording in order not to block the
			** GUI during recording time. When the record is done, it calls finished_image_record().
			*/
			std::string set_record_filename_properties(FrameDescriptor fd, std::string filename);
			void set_record();
			/*! \brief Destroy ThreadRecord cleanly */
			void finished_image_record();
			/*! \brief Browse ROI/average record output file */
			void browse_roi_output_file();
			/*! \brief Launch ROI/average record
			**
			** A ThreadCSVRecord is used for average/ROI values recording not to block
			** the GUI.
			** Close the plot if opened previously before starting the record. Call
			** finished_average_record() when done.
			*/
			void average_record();
			/*! \brief Destroy ThreadCSVRecord cleanly */
			void finished_average_record();
			/*! \brief Browse batch instruction file */
			void browse_batch_input();
			/*! \brief Configure image batch record */
			void image_batch_record();
			/*! \brief Configure average/ROI batch record */
			void csv_batch_record();
			/*! \brief Launch batch record
			**
			** Checks for errors, select which Queue to record, format path, execute
			** GPIB first block then connect to batch_next_record.
			**
			** \param path output mantissa
			*/
			void batch_record(const std::string& path);
			/*! \brief Execute next batch record
			**
			** Execute GPIB instruction and record thread alternatively until there is
			** no more GPIB instructions.
			*/
			void batch_next_record();
			/*! \brief Destroy batch record threads cleanly.
			 * Allows Qt to make the connection to the correct slot
			 * (with/without pop-up message). */
			void batch_finished_record();
			/*! \brief Destroy batch record threads cleanly.
			 *
			 * \param no_error When false, the regular pop-up message is not displayed. */
			void batch_finished_record(bool no_error);
			/*! \brief Stop image record */
			void stop_image_record();
			/*! \brief Stop average/ROI record */
			void stop_csv_record();

			/*! \brief set float output mode */
			void set_float_visible(bool value);
			/*! \brief set complex output mode */
			void set_complex_visible(bool value);

			/*! \brief Set import file src */
			void import_browse_file();
			/*! \brief Run thread_reader */
			void import_file();
			/*! \brief Stop thread_reader, and launch thread_capture */
			void import_file_stop();
			/*! \brief Update end to start if start > end */
			void import_start_spinbox_update();
			/*! \brief Update start to end if start < end */
			void import_end_spinbox_update();
			/*! \brief Hide endianess choice depending on 8/16 bit is selected*/
			void hide_endianess();
			/*! \brief change selected window */
			void change_window();

			void set_import_cine_file(bool value);
			/*! \brief Set is_cine_file value
			*/
			/*! \} */

			/*! \brief Seek import value in .cine file*/
			void seek_cine_header_data(std::string &file_src, Holovibes& holovibes);

			/* change computing state that might crash program before launching any program */
			void close_critical_compute();
			void remove_infos();
			void close_windows();

			/* reload computing values */
			void reload_ini();

			/* save computing values */
			void write_ini();

			/*! \brief Display classic GUI theme*/
			void set_classic();
			/*! \brief Display classic GUI theme*/
			void set_night();

			/*! \brief TODO a STFT signal trig*/
			void stft_signal_trig(bool checked);

			/*! \brief Detects the file properties with the title*/
			void title_detect();

			/*! \brief Rotate or flip selected window*/
			void rotateTexture();
			void flipTexture();
			#pragma endregion
			/* ---------- */
			#pragma region Protected / Private Methods
		protected:
			virtual void closeEvent(QCloseEvent* event) override;
		private:
			/*! \brief Change camera
			**
			** Delete real time OpenGL display window then destroy compute and
			** capture thread cleanly then change camera type to the given one.
			**
			** \param camera_type new camera type
			*/
			void change_camera(CameraKind c);
			/*! \brief Display error message
			** \param msg error message
			*/
			void display_error(std::string msg);
			/*! \brief Display information message
			** \param msg information message
			*/
			void display_info(std::string msg);

			/*! \brief Open a file
			** \param path file path
			*/
			void open_file(const std::string& path);
			/*! \brief Load holovibe configuration file */
			void load_ini(const std::string& path);
			/*! \brief Save holovibes configuration file */
			void save_ini(const std::string& path);
			/*! \brief Split a string each delim
			** \param str string to split
			** \param delim delimitor
			** \param elts tokens vector
			*/
			void split_string(const std::string& str, char delim, std::vector<std::string>& elts);

			void cancel_stft_view(ComputeDescriptor& cd);

			/*! \brief Format batch output file name
			**
			** Example:
			**
			** * Path: test.txt
			** * Index: 1
			** * Result: test_000001.txt
			**
			** \param path batch output file path
			** \param index index of the file
			** \return path with _index up to 10^6
			*/
			std::string format_batch_output(const std::string& path, unsigned int index);

			void	createPipe();
			void	createHoloWindow();

			#pragma endregion
			/* ---------- */
			#pragma region Fields
		private:

			enum ImportType
			{
				None,
				Camera,
				File,
			};

			std::mutex		mutex_;
			Ui::MainWindow	ui;
			Holovibes&		 holovibes_;


			/*! OpenGL windows */
			std::unique_ptr<DirectWindow>	mainDisplay;
			std::unique_ptr<SliceWindow>	sliceXZ;
			std::unique_ptr<SliceWindow>	sliceYZ;

			float		displayAngle;
			float		xzAngle;
			float		yzAngle;

			int			displayFlip;
			int			xzFlip;
			int			yzFlip;

			bool		is_enabled_autofocus_;
			bool		is_enabled_camera_;
			bool		is_enabled_average_;
			bool		is_batch_img_;
			bool		is_batch_interrupted_;
			double		z_step_;

			/*! current camera type */
			CameraKind	kCamera;
			int			import_type_;

			/*! Index of the last contrast type chosen in the affiliated QComboBox. */
			QString		last_contrast_type_;

			/*! Plot/graphic window of average/ROI computations */
			std::unique_ptr<PlotWindow> plot_window_;

			/*! Image record thread */
			std::unique_ptr<ThreadRecorder> record_thread_;
			/*! ROI/average record thread */
			std::unique_ptr<ThreadCSVRecord> CSV_record_thread_;
			/*! Number of frames to record */
			unsigned int nb_frames_;

			/*! File index used in batch recording */
			unsigned int file_index_;

			/* index used to record curent theme (0:classic 1:night)*/
			unsigned short theme_index_;

			std::shared_ptr<gpib::IVisaInterface> gpib_interface_;

			/*! \{ \name Shortcuts */
			QShortcut	*z_up_shortcut_;
			QShortcut	*z_down_shortcut_;
			QShortcut	*p_left_shortcut_;
			QShortcut	*p_right_shortcut_;
			QShortcut	*gl_full_screen_;
			QShortcut	*gl_normal_screen_;
			QShortcut	*autofocus_ctrl_c_shortcut_;
			/*! \} */
			#pragma endregion
		};
	}
}
