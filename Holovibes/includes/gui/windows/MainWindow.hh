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
* Qt main class containing the GUI. */
#pragma once

// without namespace
# include "tools.hh"
# include "json.hh"
using json = ::nlohmann::json;


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
# include "ui_mainwindow.h"
# include "ithread_input.hh"

#define GLOBAL_INI_PATH "holovibes.ini"
Q_DECLARE_METATYPE(std::function<void()>)

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
			~MainWindow();

			void notify() override;
			void notify_error(std::exception& e) override;

			DirectWindow *get_main_display();
			#pragma endregion
			/* ---------- */
			#pragma region Public Slots
		public slots:
			void on_notify();
			void update_file_reader_index(int n);
			void synchronize_thread(std::function<void()> f);
			/*! \brief Resize windows if one layout is toggled. */
			void layout_toggled();
			void configure_holovibes();
			void camera_none();
			void camera_adimec();
			void camera_edge();
			void camera_ids();
			void camera_ixon();
			void camera_hamamatsu();
			void camera_pike();
			void camera_pixelfly();
			void camera_pixelink();
			void camera_xiq();
			void camera_xib();
			void camera_photon_focus();
			void configure_camera();
			void credits();
			void documentation();
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
			void reset_input();
			void set_square_input_mode(const QString &name);
			void refreshViewMode();
			void set_convolution_mode(const bool enable);
			void set_divide_convolution_mode(const bool value);
			void toggle_renormalize(bool value);
			bool is_direct_mode();
			void reset();
			void take_reference();
			void take_sliding_ref();
			void cancel_take_reference();
			void set_filter2D();
			void set_filter2D_type(const QString &filter2Dtype);
			void cancel_filter2D();
			void setPhase();
			void set_special_buffer_size(int value);
			void update_lens_view(bool value);
			void update_raw_view(bool value);
			void set_p_accu();
			void set_x_accu();
			void set_y_accu();
			void set_composite_intervals();
			void set_composite_intervals_hsv_h_min();
			void set_composite_intervals_hsv_h_max();
			void set_composite_intervals_hsv_s_min();
			void set_composite_intervals_hsv_s_max();
			void set_composite_intervals_hsv_v_min();
			void set_composite_intervals_hsv_v_max();
			void set_composite_weights();
			void set_composite_auto_weights(bool value);
			void click_composite_rgb_or_hsv();
			void slide_update_threshold_h_min();
			void slide_update_threshold_h_max();
			void slide_update_threshold_s_min();
			void slide_update_threshold_s_max();
			void slide_update_threshold_v_min();
			void slide_update_threshold_v_max();
			void actualize_frequency_channel_s();
			void actualize_frequency_channel_v();
			void actualize_checkbox_h_gaussian_blur();
			void actualize_kernel_size_blur();
			void set_p(int value);
			void increment_p();
			void decrement_p();
			void set_wavelength(double value);
			void set_interp_lambda1(double value);
			void set_interp_lambda2(double value);
			void set_interp_sensitivity(double value);
			void set_interp_shift(int value);
			void set_interpolation(bool value);
			void set_z(double value);
			void increment_z();
			void decrement_z();
			void set_z_step(double value);
			void set_algorithm(QString value);
			void set_time_filter(QString value);
			void set_crop_stft(bool b);
			void stft_view(bool checked);
			void cancel_stft_slice_view();
			void update_stft_steps(int value);
			void set_view_mode(QString value);
			void set_unwrap_history_size(int value);
			void set_unwrapping_1d(const bool value);
			void set_unwrapping_2d(const bool value);
			void set_accumulation(bool value);
			void set_accumulation_level(int value);
			void set_xy_stabilization_enable(bool value);
			void set_xy_stabilization_show_convolution(bool value);
			void set_z_min(const double value);
			void set_z_max(const double value);
			void set_z_iter(const int value);
			void set_z_div(const int value);
			void set_autofocus_mode();
			void request_autofocus_stop();
			void pixel_size_import();
			void set_contrast_mode(bool value);
			void set_auto_contrast();
			void set_contrast_min(double value);
			void set_contrast_max(double value);
			void invert_contrast(bool value);
			void set_log_scale(bool value);
			void set_shifted_corners(bool value);
			void set_average_mode(bool value);
			void set_stabilization_area();
			void set_composite_area();
			void activeSignalZone();
			void activeNoiseZone();
			void set_average_graphic();
			void dispose_average_graphic();
			void browse_roi_file();
			void save_roi();
			void load_roi();
			void browse_convo_matrix_file();
			void load_convo_matrix();
			void browse_file();
			void set_raw_recording(bool value);
			void set_record();
			void set_record_frame_step(int value);
			void browse_roi_output_file();
			void average_record();
			void browse_batch_input();
			void image_batch_record();
			void csv_batch_record();
			void batch_record(const std::string& path);
			void batch_next_record();
			void batch_finished_record(bool no_error);
			void batch_finished_record();
			void stop_image_record();
			void finished_image_record();
			void finished_average_record();
			void stop_csv_record();
			void import_browse_file();
			void import_file();
			void import_file_stop();
			void import_start_spinbox_update();
			void import_end_spinbox_update();
			void hide_endianess();
			void change_window();
			void set_import_cine_file(bool value);
			void reload_ini();
			void write_ini();
			void set_classic();
			void set_night();
			void title_detect();
			void rotateTexture();
			void flipTexture();
			void set_scale_bar(bool value);
			void set_scale_bar_correction_factor(double value);
			void set_jitter_enabled(bool value);
			void set_jitter_factor(double value);
			void set_jitter_slices(int value);
			void set_zernike_enable(bool val);
			void set_zernike_m(int m);
			void set_zernike_n(int n);
			void set_zernike_factor(double value);
			void set_aberration_enabled(bool value);
			void set_aberration_factor(double value);
			void set_aberration_slices(int value);
			void set_square_pixel(bool value);
			void set_locked_zoom(bool value);
			void set_synchronized_record(bool value);
			void normalize(bool value);
			void to_holo_file();
			void holo_file_update();
			void display_cross(bool value);
			void reticle_scale(double value);
			//! Used to start the record at the same time than file reader.
			void start_recording();
			#pragma endregion
			/* ---------- */
		signals:
		   void synchronize_thread_signal(std::function<void()> f);
			#pragma region Protected / Private Methods
		protected:
			virtual void closeEvent(QCloseEvent* event) override;

		private:
			void 		set_direct_mode();
			void 		set_holographic_mode();
			void        set_computation_mode();
			void        set_correct_square_input_mode();
			void 		set_camera_timeout();
			void		change_camera(CameraKind c);
			void		display_error(std::string msg);
			void		display_info(std::string msg);
			void		open_file(const std::string& path);
			void		load_ini(const std::string& path);
			void		save_ini(const std::string& path);
			void		cancel_stft_view(ComputeDescriptor& cd);
			std::string	format_batch_output(const std::string& path, uint index);
			std::string	set_record_filename_properties(camera::FrameDescriptor fd, std::string filename);
			void		createPipe();
			void		createHoloWindow();
			void		close_windows();
			void		seek_cine_header_data(std::string &file_src, Holovibes& holovibes);
			void		close_critical_compute();
			void		remove_infos();
			void		pipe_refresh();
			void		set_auto_contrast_cuts();
			void		zernike_refresh();

			// Change the value without triggering any signals
			void		QSpinBoxQuietSetValue(QSpinBox* spinBox, int value);
			void		QSliderQuietSetValue(QSlider* slider, int value);
			void		QDoubleSpinBoxQuietSetValue(QDoubleSpinBox* spinBox, double value);

			// HoloFile related
			/*! Updates the import ui with the holo file settings */
			void holo_file_update_ui();
			/*! Updates the compute descriptor with the holo file settings */
			void holo_file_update_cd();
			/*! Creates a json object containing the compute descriptor settings */
			json holo_file_get_json_settings(const Queue* q);

			#pragma endregion
			/* ---------- */
			#pragma region Fields

			enum ImportType
			{
				None,
				Camera,
				File,
			};

			Ui::MainWindow		ui;
			Holovibes&			holovibes_;
			ComputeDescriptor&	compute_desc_;

			std::unique_ptr<DirectWindow>	mainDisplay;
			std::unique_ptr<SliceWindow>	sliceXZ;
			std::unique_ptr<SliceWindow>	sliceYZ;
			std::unique_ptr<DirectWindow>	lens_window;
			std::unique_ptr<DirectWindow>	raw_window;

			ushort width;
			ushort height;

			float		displayAngle;
			float		xzAngle;
			float		yzAngle;

			int			displayFlip;
			int			xzFlip;
			int			yzFlip;

			bool		is_enabled_camera_;
			bool		is_enabled_average_;
			bool		is_batch_img_;
			bool		is_batch_interrupted_;
			double		z_step_;
			unsigned	record_frame_step_;

			CameraKind	kCamera;
			ImportType	import_type_;
			QString		last_img_type_;

			std::unique_ptr<PlotWindow>				plot_window_;
			std::shared_ptr<gpib::IVisaInterface>	gpib_interface_;
			std::unique_ptr<ThreadRecorder>			record_thread_;
			std::unique_ptr<ThreadCSVRecord>		CSV_record_thread_;

			uint	nb_frames_;
			uint	file_index_;
			ushort	theme_index_;

			QShortcut	*z_up_shortcut_;
			QShortcut	*z_down_shortcut_;
			QShortcut	*p_left_shortcut_;
			QShortcut	*p_right_shortcut_;
			QShortcut	*gl_full_screen_;
			QShortcut	*gl_normal_screen_;
			QShortcut	*autofocus_ctrl_c_shortcut_;

		protected:
			void dropEvent(QDropEvent* e);
			void dragEnterEvent(QDragEnterEvent* e);
			#pragma endregion
		};
	}
}
