#include "main_window.hh"
#include "gui_gl_window.hh"
#include "gui_plot_window.hh"
#include "queue.hh"
#include "thread_recorder.hh"
#include "thread_csv_record.hh"
#include "compute_descriptor.hh"
#include "gpib_dll.hh"
#include "../GPIB/gpib_controller.hh"
#include "../GPIB/gpib_exceptions.hh"
#include "camera_exception.hh"
#include "config.hh"
#include "info_manager.hh"
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "tools.hh"
#include "gui_tool.hh"

#define GLOBAL_INI_PATH "holovibes.ini"

namespace gui
{
	MainWindow::MainWindow(holovibes::Holovibes& holovibes, QWidget *parent)
		: QMainWindow(parent)
		, holovibes_(holovibes)
		, gl_window_(nullptr)
		, gl_win_stft_0(nullptr)
		//, gl_win_stft_1(nullptr)
		, is_enabled_camera_(false)
		, is_enabled_average_(false)
		, is_batch_img_(true)
		, is_batch_interrupted_(false)
		, z_step_(0.01f)
		, camera_type_(holovibes::Holovibes::NONE)
		, last_contrast_type_("Magnitude")
		, plot_window_(nullptr)
		, record_thread_(nullptr)
		, CSV_record_thread_(nullptr)
		, file_index_(1)
		, gpib_interface_(nullptr)
		, theme_index_(0)
	{
		ui.setupUi(this);
		this->setWindowIcon(QIcon("icon1.ico"));
		InfoManager::get_manager(this->findChild<gui::GroupBox*>("Info"));

		camera_visible(false);
		record_visible(false);

		// Hide non default tab
		gui::GroupBox *special_group_box = findChild<gui::GroupBox*>("Vibrometry");
		gui::GroupBox *record_group_box = findChild<gui::GroupBox*>("Record");
		gui::GroupBox *info_group_box = findChild<gui::GroupBox*>("Info");

		QAction*      special_action = findChild<QAction*>("actionSpecial");
		QAction*      record_action = findChild<QAction*>("actionRecord");
		QAction*      info_action = findChild<QAction*>("actionInfo");

		special_action->setChecked(false);
		special_group_box->setHidden(true);
		record_action->setChecked(false);
		record_group_box->setHidden(true);
		info_action->setChecked(false);
		info_group_box->setHidden(true);

		layout_toggled(false);

		//Load Ini file
		load_ini("holovibes.ini");
		layout_toggled(false);
		if (theme_index_ == 1)
			set_night();

		// Keyboard shortcuts
		z_up_shortcut_ = new QShortcut(QKeySequence("Up"), this);
		z_up_shortcut_->setContext(Qt::ApplicationShortcut);
		connect(z_up_shortcut_, SIGNAL(activated()), this, SLOT(increment_z()));

		z_down_shortcut_ = new QShortcut(QKeySequence("Down"), this);
		z_down_shortcut_->setContext(Qt::ApplicationShortcut);
		connect(z_down_shortcut_, SIGNAL(activated()), this, SLOT(decrement_z()));

		p_left_shortcut_ = new QShortcut(QKeySequence("Left"), this);
		p_left_shortcut_->setContext(Qt::ApplicationShortcut);
		connect(p_left_shortcut_, SIGNAL(activated()), this, SLOT(decrement_p()));

		p_right_shortcut_ = new QShortcut(QKeySequence("Right"), this);
		p_right_shortcut_->setContext(Qt::ApplicationShortcut);
		connect(p_right_shortcut_, SIGNAL(activated()), this, SLOT(increment_p()));

		autofocus_ctrl_c_shortcut_ = new QShortcut(tr("Ctrl+C"), this);
		autofocus_ctrl_c_shortcut_->setContext(Qt::ApplicationShortcut);
		connect(autofocus_ctrl_c_shortcut_, SIGNAL(activated()), this, SLOT(request_autofocus_stop()));


		connect(this, SIGNAL(send_error(QString)), this, SLOT(display_message(QString)));

		QComboBox* depth_cbox = findChild<QComboBox*>("ImportDepthModeComboBox");
		connect(depth_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(hide_endianess()));

		if (is_direct_mode())
			global_visibility(false);

		/* Default size of the main window*/
		this->resize(width(), 425);
		// Display default values
		notify();
	}

	MainWindow::~MainWindow()
	{
		gl_win_stft_0.reset(nullptr);
		holovibes_.dispose_compute();
		holovibes_.dispose_capture();
		gui::InfoManager::stop_display();
	}

	void MainWindow::notify()
	{
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

		QSpinBox* STFT_step = findChild<QSpinBox*>("STFTSpinBox");
		STFT_step->setValue(cd.stft_steps.load());

		QSpinBox* phase_number = findChild<QSpinBox*>("phaseNumberSpinBox");
		phase_number->setValue(cd.nsamples);

		QSpinBox* p = findChild<QSpinBox*>("pSpinBox");
		p->setValue(cd.pindex + 1);
		p->setMaximum(cd.nsamples);

		QDoubleSpinBox* lambda = findChild<QDoubleSpinBox*>("wavelengthSpinBox");
		lambda->setValue(cd.lambda * 1.0e9f);

		QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
		z->setValue(cd.zdistance);

		QDoubleSpinBox* z_step = findChild<QDoubleSpinBox*>("zStepDoubleSpinBox");
		z_step->setValue(z_step_);

		QComboBox* algorithm = findChild<QComboBox*>("algorithmComboBox");
		if (cd.algorithm == holovibes::ComputeDescriptor::None)
			algorithm->setCurrentIndex(0);
		if (cd.algorithm == holovibes::ComputeDescriptor::FFT1)
			algorithm->setCurrentIndex(1);
		else if (cd.algorithm == holovibes::ComputeDescriptor::FFT2)
			algorithm->setCurrentIndex(2);
		else
			algorithm->setCurrentIndex(0);

		QComboBox* view_mode = findChild<QComboBox*>("viewModeComboBox");

		if (cd.view_mode == holovibes::ComputeDescriptor::MODULUS)
			view_mode->setCurrentIndex(0);
		else if (cd.view_mode == holovibes::ComputeDescriptor::SQUARED_MODULUS)
			view_mode->setCurrentIndex(1);
		else if (cd.view_mode == holovibes::ComputeDescriptor::ARGUMENT)
			view_mode->setCurrentIndex(2);
		else if (cd.view_mode == holovibes::ComputeDescriptor::PHASE_INCREASE)
			view_mode->setCurrentIndex(3);
		else if (cd.view_mode == holovibes::ComputeDescriptor::COMPLEX)
			view_mode->setCurrentIndex(4);
		else // Fallback on Modulus
			view_mode->setCurrentIndex(0);

		QCheckBox* log_scale = findChild<QCheckBox*>("logScaleCheckBox");
		log_scale->setChecked(cd.log_scale_enabled);

		QCheckBox* shift_corners = findChild<QCheckBox*>("shiftCornersCheckBox");
		shift_corners->setChecked(cd.shift_corners_enabled);

		contrast_visible(cd.contrast_enabled);

		QCheckBox* contrast = findChild<QCheckBox*>("contrastCheckBox");
		contrast->setChecked(cd.contrast_enabled);

		QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
		/* Autocontrast values depends on log_scale option. */
		if (cd.log_scale_enabled)
			contrast_min->setValue(cd.contrast_min);
		else
			contrast_min->setValue(log10(cd.contrast_min));

		QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
		if (cd.log_scale_enabled)
			contrast_max->setValue(cd.contrast_max);
		else
			contrast_max->setValue(log10(cd.contrast_max));

		QCheckBox* vibro = findChild<QCheckBox*>("vibrometryCheckBox");
		vibro->setChecked(cd.vibrometry_enabled);

		QCheckBox* convolution = findChild<QCheckBox*>("convolution_checkbox");
		if (cd.convo_matrix.size() == 0)
			convolution->setEnabled(false);
		else
			convolution->setEnabled(true);

		image_ratio_visible(cd.vibrometry_enabled);

		QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
		p_vibro->setValue(cd.pindex + 1);
		p_vibro->setMaximum(cd.nsamples);

		QSpinBox* q_vibro = findChild<QSpinBox*>("qSpinBoxVibro");
		q_vibro->setValue(cd.vibrometry_q + 1);
		q_vibro->setMaximum(cd.nsamples);

		QDoubleSpinBox* z_max = findChild<QDoubleSpinBox*>("zmaxDoubleSpinBox");
		z_max->setValue(cd.autofocus_z_max);

		QDoubleSpinBox* z_min = findChild<QDoubleSpinBox*>("zminDoubleSpinBox");
		z_min->setValue(cd.autofocus_z_min);

		QSpinBox* z_div = findChild<QSpinBox*>("zdivSpinBox");
		z_div->setValue(cd.autofocus_z_div);

		QSpinBox*  z_iter = findChild<QSpinBox*>("ziterSpinBox");
		z_iter->setValue(cd.autofocus_z_iter);

		QCheckBox* average = findChild<QCheckBox*>("averageCheckBox");
		average->setChecked(is_enabled_average_);

		GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
		if (gl_widget && is_enabled_average_ && is_direct_mode() == false)
			gl_widget->set_selection_mode(gui::eselection::AVERAGE);

		average_visible(is_enabled_average_);

		QSpinBox* special_buffer_size = findChild<QSpinBox*>("SpecialBufferSpinBox");
		special_buffer_size->setValue(cd.special_buffer_size.load());

		QSpinBox* flowgraphy_level = findChild<QSpinBox*>("FlowgraphySpinBox");
		flowgraphy_level->setValue(cd.flowgraphy_level.load());

		QCheckBox* flowgraphy_enable = findChild<QCheckBox*>("flowgraphy_checkbox");
		flowgraphy_enable->setChecked(cd.flowgraphy_enabled);

		QSpinBox* img_acc_level = findChild<QSpinBox*>("img_accSpinBox");
		img_acc_level->setValue(cd.img_acc_level.load());

		QCheckBox* img_acc_enabled = findChild<QCheckBox*>("img_accCheckBox");
		img_acc_enabled->setChecked(cd.img_acc_enabled.load());

		QDoubleSpinBox* import_pixel_size = findChild<QDoubleSpinBox*>("ImportPixelSizeDoubleSpinBox");
		import_pixel_size->setValue(cd.import_pixel_size);

		set_enable_unwrap_box();
	}

	void MainWindow::notify_error(std::exception& e, const char* msg)
	{
		holovibes::CustomException* err_ptr = dynamic_cast<holovibes::CustomException*>(&e);
		std::stringstream str;
		if (err_ptr != nullptr)
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			if (err_ptr->get_kind() == holovibes::error_kind::fail_update)
			{
				// notify will be in close_critical_compute
				if (cd.stft_enabled.load())
				{
					cd.nsamples.exchange(1);
					cd.pindex.exchange(0);
				}
				if (cd.flowgraphy_enabled.load() || cd.convolution_enabled.load())
				{
					cd.convolution_enabled.exchange(false);
					cd.flowgraphy_enabled.exchange(false);
					cd.special_buffer_size.exchange(3);
					notify();
				}
			}
			if (err_ptr->get_kind() == holovibes::error_kind::fail_accumulation)
			{
				cd.img_acc_enabled.exchange(false);
				cd.img_acc_level.exchange(1);
				notify();
			}
			close_critical_compute();

			str << "GPU allocation error occured." << std::endl << "Cuda error message: " << std::endl << msg;
			emit send_error(QString::fromLatin1(str.str().c_str()));
		}
		else
		{
			str << "Unknown error occured.";
			emit send_error(QString::fromLatin1(str.str().c_str()));
		}
	}

	void MainWindow::display_message(QString msg)
	{
		QMessageBox msg_box(0);
		msg_box.setText(msg);
		msg_box.setIcon(QMessageBox::Critical);
		msg_box.exec();
	}

	void MainWindow::layout_toggled(bool b)
	{
		unsigned int childCount = 0;
		std::vector<gui::GroupBox*> v;

		v.push_back(findChild<gui::GroupBox*>("ImageRendering"));
		v.push_back(findChild<gui::GroupBox*>("View"));
		v.push_back(findChild<gui::GroupBox*>("Vibrometry"));
		v.push_back(findChild<gui::GroupBox*>("Record"));
		v.push_back(findChild<gui::GroupBox*>("Import"));
		v.push_back(findChild<gui::GroupBox*>("Info"));

		for each (gui::GroupBox* var in v)
			childCount += !var->isHidden();

		if (childCount > 0)
			this->resize(QSize(childCount * 195, 425));
		else
			this->resize(QSize(195, 60));
	}

	void MainWindow::configure_holovibes()
	{
		open_file(holovibes_.get_launch_path() + "/" + GLOBAL_INI_PATH);
	}

	void MainWindow::gl_full_screen()
	{
		if (gl_window_)
			gl_window_->full_screen();
		else
			display_error("No camera selected");
	}

	void MainWindow::camera_ids()
	{
		change_camera(holovibes::Holovibes::IDS);
	}

	void MainWindow::camera_ixon()
	{
		change_camera(holovibes::Holovibes::IXON);
	}

	void MainWindow::camera_none()
	{
		gl_window_.reset(nullptr);
		if (!is_direct_mode())
			holovibes_.dispose_compute();
		holovibes_.dispose_capture();
		camera_visible(false);
		record_visible(false);
		global_visibility(false);

		QAction* settings = findChild<QAction*>("actionSettings");
		settings->setEnabled(false);
	}

	void MainWindow::camera_adimec()
	{
		change_camera(holovibes::Holovibes::ADIMEC);
	}

	void MainWindow::camera_edge()
	{
		change_camera(holovibes::Holovibes::EDGE);
	}

	void MainWindow::camera_pike()
	{
		change_camera(holovibes::Holovibes::PIKE);
	}

	void MainWindow::camera_pixelfly()
	{
		change_camera(holovibes::Holovibes::PIXELFLY);
	}

	void MainWindow::camera_xiq()
	{
		change_camera(holovibes::Holovibes::XIQ);
	}

	void MainWindow::credits()
	{
		display_info("Holovibes " + holovibes::version + "\n\n"

			"Developers:\n"
			"Cyril Cetre\n"
			"Cl�ment Ledant\n"

			"Eric Delanghe\n"
			"Arnaud Gaillard\n"
			"Geoffrey Le Gourri�rec\n"

			"Jeffrey Bencteux\n"
			"Thomas Kostas\n"
			"Pierre Pagnoux\n"

			"Antoine Dill�e\n"
			"Romain Cancilli�re\n"

			"Michael Atlan\n");
	}

	void MainWindow::configure_camera()
	{
		open_file(boost::filesystem::current_path().generic_string() + "/" + holovibes_.get_camera_ini_path());
	}

	void MainWindow::init_image_mode(QPoint& pos, unsigned int& width, unsigned int& height)
	{
		holovibes_.dispose_compute();

		if (gl_window_)
		{
			pos = gl_window_->pos();
			width = gl_window_->size().width();
			height = gl_window_->size().height();
		}

		if (gl_window_)
			gl_window_.reset(nullptr);
	}

	void MainWindow::set_direct_mode()
	{
		close_critical_compute();
		if (is_enabled_camera_)
		{
			holovibes_.get_compute_desc().compute_mode = holovibes::ComputeDescriptor::compute_mode::DIRECT;
			QPoint pos(0, 0);
			unsigned int width = 512;
			unsigned int height = 512;
			init_image_mode(pos, width, height);
			gl_window_.reset(new GuiGLWindow(pos, width, height, holovibes_, holovibes_.get_capture_queue()));
			set_convolution_mode(false);
			global_visibility(false);
			notify();
		}
	}

	void MainWindow::set_holographic_mode()
	{
		close_critical_compute();
		if (is_enabled_camera_)
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.compute_mode = holovibes::ComputeDescriptor::compute_mode::HOLOGRAM;
			QPoint pos(0, 0);
			unsigned int width = 512;
			unsigned int height = 512;
			init_image_mode(pos, width, height);
			unsigned int depth = 2;
			try
			{
				if (cd.view_mode == holovibes::ComputeDescriptor::COMPLEX)
				{
					last_contrast_type_ = "Complex output";
					depth = 8;
				}
				holovibes_.init_compute(holovibes::ThreadCompute::PipeType::PIPE, depth);
				holovibes_.get_pipe()->register_observer(*this);
				gl_window_.reset(new GuiGLWindow(pos, width, height, holovibes_, holovibes_.get_output_queue()));
				if (!cd.flowgraphy_enabled && !is_direct_mode())
					holovibes_.get_pipe()->request_autocontrast();
				global_visibility(true);
			}
			catch (std::exception& e)
			{
				display_error(e.what());
			}
			notify();
		}
	}

	void MainWindow::set_complex_mode(bool value)
	{
		close_critical_compute();
		QPoint pos(0, 0);
		unsigned int width = 512;
		unsigned int height = 512;
		init_image_mode(pos, width, height);
		unsigned int depth = 2;
		try
		{
			if (value == true)
				depth = 8;
			holovibes_.init_compute(holovibes::ThreadCompute::PipeType::PIPE, depth);
			holovibes_.get_pipe()->register_observer(*this);
			//global_visibility(true);
			gl_window_.reset(new GuiGLWindow(pos, width, height, holovibes_, holovibes_.get_output_queue()));
		}
		catch (std::exception& e)
		{
			display_error(e.what());
		}
	}

	void MainWindow::set_convolution_mode(const bool value)
	{
		QCheckBox* convo = findChild<QCheckBox*>("convolution_checkbox");

		if (value == true && holovibes_.get_compute_desc().convo_matrix.empty())
		{
			convo->setChecked(false);
			display_error("No valid kernel has been given");
			holovibes_.get_compute_desc().convolution_enabled = false;

		}
		else
		{
			convo->setChecked(value);
			holovibes_.get_compute_desc().convolution_enabled = value;
			if (!holovibes_.get_compute_desc().flowgraphy_enabled && !is_direct_mode())
				holovibes_.get_pipe()->request_autocontrast();
		}
	}

	void MainWindow::set_flowgraphy_mode(const bool value)
	{
		holovibes_.get_compute_desc().flowgraphy_enabled = value;
		if (!is_direct_mode())
			holovibes_.get_pipe()->request_refresh();
	}

	bool MainWindow::is_direct_mode()
	{
		return holovibes_.get_compute_desc().compute_mode == holovibes::ComputeDescriptor::compute_mode::DIRECT;
	}

	void MainWindow::set_image_mode()
	{
		if (holovibes_.get_compute_desc().compute_mode == holovibes::ComputeDescriptor::compute_mode::DIRECT)
			set_direct_mode();
		if (holovibes_.get_compute_desc().compute_mode == holovibes::ComputeDescriptor::compute_mode::HOLOGRAM)
			set_holographic_mode();
	}

	void MainWindow::reset()
	{
		holovibes::Config&	config = global::global_config;
		int					device = 0;

		global_visibility(false);
		auto manager = gui::InfoManager::get_manager();
		manager->update_info("Status", "Resetting...");
		qApp->processEvents();
		gl_window_.reset(nullptr);
		if (!is_direct_mode())
			holovibes_.dispose_compute();
		holovibes_.dispose_capture();
		camera_visible(false);
		record_visible(false);
		if (config.set_cuda_device == 1)
		{
			if (config.auto_device_number == 1)
			{
				cudaGetDevice(&device);
				config.device_number = device;
			}
			else
				device = config.device_number;
			cudaSetDevice(device);
		}
		cudaDeviceSynchronize();
		cudaDeviceReset();
		change_camera(camera_type_);
		load_ini("holovibes.ini");
		manager->remove_info("Status");
	}

	void MainWindow::take_reference()
	{
		if (!is_direct_mode())
		{
			QPushButton* cancel = findChild<QPushButton*>("cancelrefPushButton");
			cancel->setEnabled(true);
			QPushButton* sliding = findChild<QPushButton*>("slindingrefPushButton");
			sliding->setEnabled(false);
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.ref_diff_enabled.exchange(true);
			holovibes_.get_pipe()->request_ref_diff_refresh();
			gui::InfoManager::update_info_safe("Reference", "Processing... ");
		}
	}

	void MainWindow::take_sliding_ref()
	{
		if (!is_direct_mode())
		{
			QPushButton* cancel = findChild<QPushButton*>("cancelrefPushButton");
			cancel->setEnabled(true);
			QPushButton* takeref = findChild<QPushButton*>("takerefPushButton");
			takeref->setEnabled(false);
			QPushButton* sliding = findChild<QPushButton*>("slindingrefPushButton");
			sliding->setEnabled(false);
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.ref_sliding_enabled.exchange(true);
			holovibes_.get_pipe()->request_ref_diff_refresh();
			gui::InfoManager::update_info_safe("Reference", "Processing...");
		}
	}

	void MainWindow::cancel_take_reference()
	{
		if (!is_direct_mode())
		{
			QPushButton* cancel = findChild<QPushButton*>("cancelrefPushButton");
			cancel->setEnabled(false);
			QPushButton* sliding = findChild<QPushButton*>("slindingrefPushButton");
			sliding->setEnabled(true);
			QPushButton* takeref = findChild<QPushButton*>("takerefPushButton");
			takeref->setEnabled(true);
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.ref_diff_enabled.exchange(false);
			cd.ref_sliding_enabled.exchange(false);
			holovibes_.get_pipe()->request_ref_diff_refresh();
			gui::InfoManager::remove_info_safe("Reference");
		}
	}

	void MainWindow::set_filter2D()
	{
		if (!is_direct_mode())
		{
			QPushButton* cancel = findChild<QPushButton*>("cancelFilter2DPushButton");
			cancel->setEnabled(true);
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
			gl_widget->set_selection_mode(gui::eselection::STFT_ROI);
			connect(gl_widget, SIGNAL(stft_roi_zone_selected_update(holovibes::Rectangle)), this, SLOT(request_stft_roi_update(holovibes::Rectangle)),
				Qt::UniqueConnection);
			connect(gl_widget, SIGNAL(stft_roi_zone_selected_end()), this, SLOT(request_stft_roi_end()),
				Qt::UniqueConnection);
			cd.log_scale_enabled.exchange(true);
			cd.shift_corners_enabled.exchange(false);
			if (cd.contrast_enabled)
			{
				QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
				QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
				set_contrast_min(contrast_min->value());
				set_contrast_max(contrast_max->value());
			}
			cd.filter_2d_enabled.exchange(true);
			// notify();
			holovibes_.get_pipe()->request_autocontrast();
			gui::InfoManager::update_info_safe("Filter2D", "Processing...");
		}
	}

	void MainWindow::cancel_filter2D()
	{
		if (!is_direct_mode())
		{
			QPushButton* cancel = findChild<QPushButton*>("cancelFilter2DPushButton");
			cancel->setEnabled(false);
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
			gl_widget->set_selection_mode(gui::eselection::ZOOM);
			cd.filter_2d_enabled.exchange(false);
			cd.stft_roi_zone.exchange(holovibes::Rectangle(holovibes::Point2D(0, 0), holovibes::Point2D(0, 0)));
			gui::InfoManager::remove_info_safe("Filter2D");
			holovibes_.get_pipe()->request_autocontrast();
		}
	}

	void MainWindow::set_phase_number(const int value)
	{
		holovibes::Queue* input;
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			input = &holovibes_.get_capture_queue();
			if (cd.stft_enabled || (value <= static_cast<const int>(input->get_max_elts())))
			{
				holovibes_.get_pipe()->request_update_n(value);
				notify();
			}
			else
			{
				QSpinBox* p_nphase = findChild<QSpinBox*>("phaseNumberSpinBox");
				p_nphase->setValue(value - 1);
			}
		}
	}

	void MainWindow::set_special_buffer_size(int value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.special_buffer_size.exchange(value);
			if (cd.special_buffer_size < static_cast<std::atomic<int>>(cd.flowgraphy_level))
			{
				if (cd.special_buffer_size % 2 == 0)
					cd.flowgraphy_level = cd.special_buffer_size - 1;
				else
					cd.flowgraphy_level = cd.special_buffer_size;
			}
			notify();
			if (!holovibes_.get_compute_desc().flowgraphy_enabled)
				holovibes_.get_pipe()->request_autocontrast();
		}
	}

	void MainWindow::set_p(int value)
	{
		value--;
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (value < static_cast<int>(cd.nsamples))
			{
				// Synchronize with p_vibro
				QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
				p_vibro->setValue(value + 1);

				cd.pindex.exchange(value);
				if (!holovibes_.get_compute_desc().flowgraphy_enabled && !is_direct_mode())
					holovibes_.get_pipe()->request_autocontrast();
			}
			else
				display_error("p param has to be between 0 and n");
		}
	}

	void MainWindow::set_flowgraphy_level(const int value)
	{
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
		int flag = 0;

		if (!is_direct_mode())
		{
			if (value % 2 == 0)
			{
				if (value + 1 <= cd.special_buffer_size)
				{
					cd.flowgraphy_level.exchange(value + 1);
					flag = 1;
				}
			}
			else
			{
				if (value <= cd.special_buffer_size)
				{
					cd.flowgraphy_level.exchange(value);
					flag = 1;
				}
			}
			notify();
			if (flag == 1)
				holovibes_.get_pipe()->request_refresh();
		}
	}

	void MainWindow::increment_p()
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (cd.pindex < cd.nsamples)
			{
				++(cd.pindex);
				notify();
				if (!holovibes_.get_compute_desc().flowgraphy_enabled)
					holovibes_.get_pipe()->request_autocontrast();
			}
			else
				display_error("p param has to be between 1 and n");
		}
	}

	void MainWindow::decrement_p()
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (cd.pindex > 0)
			{
				--(cd.pindex);
				notify();
				if (!holovibes_.get_compute_desc().flowgraphy_enabled)
					holovibes_.get_pipe()->request_autocontrast();
			}
			else
				display_error("p param has to be between 1 and n");
		}
	}

	void MainWindow::set_wavelength(const double value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.lambda = static_cast<float>(value)* 1.0e-9f;
			holovibes_.get_pipe()->request_refresh();

			// Updating the GUI
			QLineEdit* boundary = findChild<QLineEdit*>("boundary");
			boundary->clear();
			boundary->insert(QString::number(holovibes_.get_boundary()));
		}
	}

	void MainWindow::set_z(const double value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.zdistance = static_cast<float>(value);
			holovibes_.get_pipe()->request_refresh();
		}
	}

	void MainWindow::increment_z()
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			set_z(cd.zdistance + z_step_);
			QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
			z->setValue(cd.zdistance);
		}
	}

	void MainWindow::decrement_z()
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			set_z(cd.zdistance - z_step_);
			QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
			z->setValue(cd.zdistance);
		}
	}

	void MainWindow::set_z_step(const double value)
	{
		z_step_ = value;
		QDoubleSpinBox* z_spinbox = findChild<QDoubleSpinBox*>("zSpinBox");
		z_spinbox->setSingleStep(value);
	}

	void MainWindow::set_algorithm(const QString value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			//QSpinBox* phaseNumberSpinBox = findChild<QSpinBox*>("phaseNumberSpinBox");
			GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
			gl_widget->set_selection_mode(gui::eselection::ZOOM);
			if (value == "None")
				cd.algorithm = holovibes::ComputeDescriptor::None;
			else if (value == "1FFT")
				cd.algorithm = holovibes::ComputeDescriptor::FFT1;
			else if (value == "2FFT")
				cd.algorithm = holovibes::ComputeDescriptor::FFT2;
			else
				assert(!"Unknow Algorithm.");

			if (!holovibes_.get_compute_desc().flowgraphy_enabled)
				holovibes_.get_pipe()->request_autocontrast();
		}
	}

	void MainWindow::set_stft(bool b)
	{
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
		if (!is_direct_mode())
		{
			unsigned int tmp = cd.nsamples.load();
			cd.nsamples.exchange(cd.stft_level.load());
			cd.stft_level.exchange(tmp);
			cd.stft_enabled = b;
			holovibes_.get_pipe()->request_update_n(cd.nsamples);
			notify();
			QCheckBox* p = findChild<QCheckBox*>("stft_view_checkbox");
			p->setEnabled((b) ? true : false);
		}
	}

	void MainWindow::update_stft_steps(int value)
	{
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
		if (!is_direct_mode())
		{
			cd.stft_steps.exchange(value);
		}
	}

	void MainWindow::set_view_mode(const QString value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

			// Reenabling phase number and p adjustments.
			QSpinBox* phase_number = findChild<QSpinBox*>("phaseNumberSpinBox");
			phase_number->setEnabled(true);

			QSpinBox* p = findChild<QSpinBox*>("pSpinBox");
			p->setEnabled(true);

			// QCheckBox* pipeline_checkbox = findChild<QCheckBox*>("PipelineCheckBox");
			bool pipeline_checked = false; //pipeline_checkbox->isChecked();

			std::cout << "Value = " << value.toUtf8().constData() << std::endl;
			if (last_contrast_type_ == "Complex output" && value != "Complex output")
			{
				set_complex_mode(false);
			}
			if (value == "Magnitude")
			{
				cd.view_mode = holovibes::ComputeDescriptor::MODULUS;
				last_contrast_type_ = value;
			}
			else if (value == "Squared magnitude")
			{
				cd.view_mode = holovibes::ComputeDescriptor::SQUARED_MODULUS;
				last_contrast_type_ = value;
			}
			else if (value == "Argument")
			{
				cd.view_mode = holovibes::ComputeDescriptor::ARGUMENT;
				last_contrast_type_ = value;
			}
			else if (value == "Complex output")
			{
				set_complex_mode(true);
				cd.view_mode = holovibes::ComputeDescriptor::COMPLEX;
				last_contrast_type_ = value;
			}
			else
			{
				if (pipeline_checked)
				{
					// For now, phase unwrapping is only usable with the Pipe, not the Pipeline.
					display_error("Unwrapping is not available with the Pipeline.");
					QComboBox* contrast_type = findChild<QComboBox*>("viewModeComboBox");
					// last_contrast_type_ exists for this sole purpose...
					contrast_type->setCurrentIndex(contrast_type->findText(last_contrast_type_));
				}
				else
				{
					if (value == "Phase increase")
						cd.view_mode = holovibes::ComputeDescriptor::PHASE_INCREASE;
				}
			}
			if (!holovibes_.get_compute_desc().flowgraphy_enabled)
				holovibes_.get_pipe()->request_autocontrast();

			set_enable_unwrap_box();
		}
	}

	void MainWindow::set_unwrap_history_size(int value)
	{
		if (!is_direct_mode())
		{
			holovibes_.get_compute_desc().unwrap_history_size = value;
			holovibes_.get_pipe()->request_update_unwrap_size(value);
		}
	}

	void MainWindow::set_unwrapping_1d(const bool value)
	{
		if (!is_direct_mode())
		{
			auto pipe = holovibes_.get_pipe();
			pipe->request_unwrapping_1d(value);
			pipe->request_refresh();
		}
	}

	void MainWindow::set_unwrapping_2d(const bool value)
	{
		if (!is_direct_mode())
		{
			auto pipe = holovibes_.get_pipe();
			pipe->request_unwrapping_2d(value);
			pipe->request_refresh();
		}
	}

	void MainWindow::set_enable_unwrap_box()
	{
		QCheckBox* unwrap_2d = findChild<QCheckBox*>("unwrap_2d_CheckBox");
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

		if ((cd.view_mode == holovibes::ComputeDescriptor::PHASE_INCREASE) |
			(cd.view_mode == holovibes::ComputeDescriptor::ARGUMENT))
		{
			unwrap_2d->setEnabled(true);
		}
		else
			unwrap_2d->setEnabled(false);
	}

	void MainWindow::set_accumulation(bool value)
	{
		if (!is_direct_mode())
		{
			holovibes_.get_compute_desc().img_acc_enabled.exchange(value);
			// if (!value)
			holovibes_.get_pipe()->request_acc_refresh();
			// else
			//  holovibes_.get_pipe()->request_refresh();
		}
	}

	void MainWindow::set_accumulation_level(int value)
	{
		if (!is_direct_mode())
		{
			holovibes_.get_compute_desc().img_acc_level.exchange(value);
			holovibes_.get_pipe()->request_acc_refresh();
		}
		notify();
	}

	void MainWindow::set_z_min(const double value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.autofocus_z_min = value;
		}
	}

	void MainWindow::set_z_max(const double value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.autofocus_z_max = value;
		}
	}

	void MainWindow::set_import_pixel_size(const double value)
	{
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
		cd.import_pixel_size = value;
	}

	void MainWindow::set_z_iter(const int value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.autofocus_z_iter = value;
		}
	}

	void MainWindow::set_z_div(const int value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.autofocus_z_div = value;
		}
	}

	void MainWindow::set_autofocus_mode()
	{
		GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
		gl_widget->set_selection_mode(gui::eselection::AUTOFOCUS);

		const float z_max = findChild<QDoubleSpinBox*>("zmaxDoubleSpinBox")->value();
		const float z_min = findChild<QDoubleSpinBox*>("zminDoubleSpinBox")->value();
		const unsigned int z_div = findChild<QSpinBox*>("zdivSpinBox")->value();
		const unsigned int z_iter = findChild<QSpinBox*>("ziterSpinBox")->value();
		holovibes::ComputeDescriptor& desc = holovibes_.get_compute_desc();

		if (desc.stft_enabled)
		{
			display_error("You can't call autofocus in stft mode.");
			return;
		}

		if (z_min < z_max)
		{
			desc.autofocus_z_min = z_min;
			desc.autofocus_z_max = z_max;
			desc.autofocus_z_div.exchange(z_div);
			desc.autofocus_z_iter = z_iter;

			connect(gl_widget, SIGNAL(autofocus_zone_selected(holovibes::Rectangle)), this, SLOT(request_autofocus(holovibes::Rectangle)),
				Qt::UniqueConnection);
		}
		else
			display_error("z min has to be strictly inferior to z max");
	}

	void MainWindow::request_autofocus(holovibes::Rectangle zone)
	{
		auto manager = gui::InfoManager::get_manager();
		manager->update_info("Status", "Autofocus processing...");
		GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
		holovibes::ComputeDescriptor& desc = holovibes_.get_compute_desc();

		desc.autofocus_zone = zone;
		holovibes_.get_pipe()->request_autofocus();
		gl_widget->set_selection_mode(gui::eselection::ZOOM);
	}

	void MainWindow::request_stft_roi_end()
	{
		holovibes_.get_pipe()->request_filter2D_roi_end();
	}

	void MainWindow::request_stft_roi_update(holovibes::Rectangle zone)
	{
		//GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
		holovibes::ComputeDescriptor& desc = holovibes_.get_compute_desc();

		desc.stft_roi_zone = zone;
		holovibes_.get_pipe()->request_filter2D_roi_update();
	}

	void MainWindow::request_autofocus_stop()
	{
		try
		{
			holovibes_.get_pipe()->request_autofocus_stop();
		}
		catch (std::runtime_error& e)
		{
			std::cerr << e.what() << std::endl;
		}
	}

	void MainWindow::set_contrast_mode(bool value)
	{
		QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
		QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
		contrast_visible(value);

		if (!is_direct_mode())
		{
			holovibes_.get_compute_desc().contrast_enabled.exchange(value);

			set_contrast_min(contrast_min->value());
			set_contrast_max(contrast_max->value());

			holovibes_.get_pipe()->request_refresh();
		}
	}

	void MainWindow::set_auto_contrast()
	{
		if (!is_direct_mode())
			holovibes_.get_pipe()->request_autocontrast();
	}

	void MainWindow::set_contrast_min(const double value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (cd.contrast_enabled)
			{
				if (cd.log_scale_enabled)
					cd.contrast_min = value;
				else
					cd.contrast_min = pow(10, value);

				holovibes_.get_pipe()->request_refresh();
			}
		}
	}

	void MainWindow::set_contrast_max(const double value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (cd.contrast_enabled)
			{
				if (cd.log_scale_enabled)
					cd.contrast_max = value;
				else
					cd.contrast_max = pow(10, value);

				holovibes_.get_pipe()->request_refresh();
			}
		}
	}

	void MainWindow::set_log_scale(const bool value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.log_scale_enabled.exchange(value);

			if (cd.contrast_enabled)
			{
				QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
				QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
				set_contrast_min(contrast_min->value());
				set_contrast_max(contrast_max->value());
			}

			holovibes_.get_pipe()->request_refresh();
		}
	}

	void MainWindow::set_shifted_corners(const bool value)
	{
		if (!is_direct_mode())
		{
			holovibes_.get_compute_desc().shift_corners_enabled.exchange(value);
			holovibes_.get_pipe()->request_refresh();
		}
	}

	void MainWindow::set_vibro_mode(const bool value)
	{
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
			if (cd.pindex.load() > cd.nsamples)
				cd.pindex.exchange(cd.nsamples);
			if (cd.vibrometry_q.load() > cd.nsamples)
				cd.vibrometry_q.exchange(cd.nsamples);
			image_ratio_visible(value);
			cd.vibrometry_enabled.exchange(value);
			holovibes_.get_pipe()->request_refresh();
			notify();
		}
	}

	void MainWindow::set_p_vibro(int value)
	{
		value--;
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (value < static_cast<int>(cd.nsamples) && value >= 0)
			{
				cd.pindex.exchange(value);
				notify();
				holovibes_.get_pipe()->request_refresh();
			}
			else
				display_error("p param has to be between 1 and n");;
		}
	}

	void MainWindow::set_q_vibro(int value)
	{
		value--;
		if (!is_direct_mode())
		{
			holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (value < static_cast<int>(cd.nsamples) && value >= 0)
			{
				holovibes_.get_compute_desc().vibrometry_q.exchange(value);
				holovibes_.get_pipe()->request_refresh();
			}
			else
				display_error("q param has to be between 1 and phase #");
		}
	}

	void MainWindow::set_average_mode(const bool value)
	{
		GLWidget * gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
		if (value)
			gl_widget->set_selection_mode(gui::eselection::AVERAGE);
		else
			gl_widget->set_selection_mode(gui::eselection::ZOOM);
		is_enabled_average_ = value;

		average_visible(value);
	}

	void MainWindow::set_average_graphic()
	{
		PlotWindow* plot_window = new PlotWindow(holovibes_.get_average_queue(), "ROI Average");

		connect(plot_window, SIGNAL(closed()), this, SLOT(dispose_average_graphic()), Qt::UniqueConnection);
		holovibes_.get_pipe()->request_average(&holovibes_.get_average_queue());
		holovibes_.get_pipe()->request_refresh();
		plot_window_.reset(plot_window);
	}

	void MainWindow::dispose_average_graphic()
	{
		plot_window_.reset(nullptr);
		holovibes_.get_pipe()->request_average_stop();
		holovibes_.get_average_queue().clear();
		holovibes_.get_pipe()->request_refresh();
	}

	void MainWindow::browse_roi_file()
	{
		QString filename = QFileDialog::getSaveFileName(this,
			tr("ROI output file"), "C://", tr("Ini files (*.ini)"));

		QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
		roi_output_line_edit->clear();
		roi_output_line_edit->insert(filename);
	}

	void MainWindow::browse_convo_matrix_file()
	{
		QString filename = QFileDialog::getOpenFileName(this,
			tr("Matrix file"), "C://", tr("Txt files (*.txt)"));

		QLineEdit* matrix_output_line_edit = findChild<QLineEdit*>("ConvoMatrixLineEdit");
		matrix_output_line_edit->clear();
		matrix_output_line_edit->insert(filename);
	}

	void MainWindow::browse_roi_output_file()
	{
		QString filename = QFileDialog::getSaveFileName(this,
			tr("ROI output file"), "C://", tr("Text files (*.txt);;CSV files (*.csv)"));

		QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("ROIOutputLineEdit");
		roi_output_line_edit->clear();
		roi_output_line_edit->insert(filename);
	}

	void MainWindow::save_roi()
	{
		QLineEdit* path_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
		std::string path = path_line_edit->text().toUtf8();

		if (path != "")
		{
			boost::property_tree::ptree ptree;
			const GLWidget& gl_widget = gl_window_->get_gl_widget();
			const holovibes::Rectangle& signal = gl_widget.get_signal_selection();
			const holovibes::Rectangle& noise = gl_widget.get_noise_selection();

			ptree.put("signal.top_left_x", signal.top_left.x);
			ptree.put("signal.top_left_y", signal.top_left.y);
			ptree.put("signal.bottom_right_x", signal.bottom_right.x);
			ptree.put("signal.bottom_right_y", signal.bottom_right.y);

			ptree.put("noise.top_left_x", noise.top_left.x);
			ptree.put("noise.top_left_y", noise.top_left.y);
			ptree.put("noise.bottom_right_x", noise.bottom_right.x);
			ptree.put("noise.bottom_right_y", noise.bottom_right.y);

			boost::property_tree::write_ini(path, ptree);
			display_info("Roi saved in " + path);
		}
		else
			display_error("Invalid path");
	}

	void MainWindow::load_roi()
	{
		QLineEdit* path_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
		const std::string path = path_line_edit->text().toUtf8();
		boost::property_tree::ptree ptree;
		GLWidget& gl_widget = gl_window_->get_gl_widget();

		try
		{
			boost::property_tree::ini_parser::read_ini(path, ptree);

			holovibes::Point2D signal_top_left;
			holovibes::Point2D signal_bottom_right;
			holovibes::Point2D noise_top_left;
			holovibes::Point2D noise_bottom_right;

			signal_top_left.x = ptree.get<int>("signal.top_left_x", 0);
			signal_top_left.y = ptree.get<int>("signal.top_left_y", 0);
			signal_bottom_right.x = ptree.get<int>("signal.bottom_right_x", 0);
			signal_bottom_right.y = ptree.get<int>("signal.bottom_right_y", 0);

			noise_top_left.x = ptree.get<int>("noise.top_left_x", 0);
			noise_top_left.y = ptree.get<int>("noise.top_left_y", 0);
			noise_bottom_right.x = ptree.get<int>("noise.bottom_right_x", 0);
			noise_bottom_right.y = ptree.get<int>("noise.bottom_right_y", 0);

			holovibes::Rectangle signal(signal_top_left, signal_bottom_right);
			holovibes::Rectangle noise(noise_top_left, noise_bottom_right);

			gl_widget.set_signal_selection(signal);
			gl_widget.set_noise_selection(noise);
			gl_widget.enable_selection();
		}
		catch (std::exception& e)
		{
			display_error("Couldn't load ini file\n" + std::string(e.what()));
		}
	}

	void MainWindow::load_convo_matrix()
	{
		QLineEdit* path_line_edit = findChild<QLineEdit*>("ConvoMatrixLineEdit");
		const std::string path = path_line_edit->text().toUtf8();
		boost::property_tree::ptree ptree;
		//GLWidget& gl_widget = gl_window_->get_gl_widget();
		holovibes::ComputeDescriptor& desc = holovibes_.get_compute_desc();
		std::stringstream strStream;
		std::string str;
		std::string delims = " \f\n\r\t\v";
		std::vector<std::string> v_str;
		std::vector<std::string> matrix_size;
		std::vector<std::string> matrix;
		//QCheckBox* convo = findChild<QCheckBox*>("convolution_checkbox");
		set_convolution_mode(false);
		holovibes_.reset_convolution_matrix();

		try
		{
			std::ifstream file(path);
			unsigned int c = 0;

			strStream << file.rdbuf();
			file.close();
			str = strStream.str();
			boost::split(v_str, str, boost::is_any_of(";"));
			if (v_str.size() != 2)
			{
				display_error("Couldn't load file : too much or to little separator\n");
				notify();
				return;
			}
			boost::trim(v_str[0]);
			boost::split(matrix_size, v_str[0], boost::is_any_of(delims), boost::token_compress_on);
			if (matrix_size.size() != 3)
			{
				display_error("Couldn't load file : too much or too little arguments for size\n");
				notify();
				return;
			}
			desc.convo_matrix_width = std::stoi(matrix_size[0]);
			desc.convo_matrix_height = std::stoi(matrix_size[1]);
			desc.convo_matrix_z = std::stoi(matrix_size[2]);
			boost::trim(v_str[1]);
			boost::split(matrix, v_str[1], boost::is_any_of(delims), boost::token_compress_on);
			while (c < matrix.size())
			{
				if (matrix[c] != "")
					desc.convo_matrix.push_back(std::stof(matrix[c]));
				c++;
			}
			if ((desc.convo_matrix_width * desc.convo_matrix_height * desc.convo_matrix_z) != matrix.size())
			{
				holovibes_.reset_convolution_matrix();
				display_error("Couldn't load file : invalid file\n");
			}
		}
		catch (std::exception& e)
		{
			holovibes_.reset_convolution_matrix();
			display_error("Couldn't load file\n" + std::string(e.what()));
		}
		notify();
	}

	void MainWindow::browse_file()
	{
		QString filename = QFileDialog::getSaveFileName(this,
			tr("Record output file"), "C://", tr("Raw files (*.raw);; All files (*)"));

		QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
		path_line_edit->clear();
		path_line_edit->insert(filename);
	}

	void MainWindow::set_record()
	{
		global_visibility(false);
		record_but_cancel_visible(false);

		QSpinBox*  nb_of_frames_spinbox = findChild<QSpinBox*>("numberOfFramesSpinBox");
		QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
		QCheckBox* float_output_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");
		QCheckBox* complex_output_checkbox = findChild<QCheckBox*>("RecordComplexOutputCheckBox");

		int nb_of_frames = nb_of_frames_spinbox->value();
		std::string path = path_line_edit->text().toUtf8();
		holovibes::Queue* queue;

		try
		{
			if (float_output_checkbox->isChecked() && !is_direct_mode())
			{
				std::shared_ptr<holovibes::ICompute> pipe = holovibes_.get_pipe();
				camera::FrameDescriptor frame_desc = holovibes_.get_output_queue().get_frame_desc();

				frame_desc.depth = sizeof (float);
				queue = new holovibes::Queue(frame_desc, global::global_config.float_queue_max_size, "FloatQueue");
				pipe->request_float_output(queue);
			}
			else if (complex_output_checkbox->isChecked() && !is_direct_mode())
			{
				std::shared_ptr<holovibes::ICompute> pipe = holovibes_.get_pipe();
				camera::FrameDescriptor frame_desc = holovibes_.get_output_queue().get_frame_desc();

				frame_desc.depth = sizeof (cufftComplex);
				queue = new holovibes::Queue(frame_desc, global::global_config.float_queue_max_size, "ComplexQueue");
				pipe->request_complex_output(queue);
			}
			else if (is_direct_mode())
				queue = &holovibes_.get_capture_queue();
			else
				queue = &holovibes_.get_output_queue();

			record_thread_.reset(new ThreadRecorder(
				*queue,
				path,
				nb_of_frames,
				this));

			connect(record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_image_record()));
			record_thread_->start();

			QPushButton* cancel_button = findChild<QPushButton*>("cancelPushButton");
			cancel_button->setDisabled(false);
		}
		catch (std::exception& e)
		{
			display_error(e.what());
			global_visibility(true);
			record_but_cancel_visible(true);
		}
	}

	void MainWindow::finished_image_record()
	{
		QCheckBox* float_output_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");
		QCheckBox* complex_output_checkbox = findChild<QCheckBox*>("RecordComplexOutputCheckBox");
		QProgressBar*   progress_bar = InfoManager::get_manager()->get_progress_bar();

		record_thread_.reset(nullptr);
		display_info("Record done");
		progress_bar->setMaximum(1);
		progress_bar->setValue(1);
		if (float_output_checkbox->isChecked() && !is_direct_mode())
			holovibes_.get_pipe()->request_float_output_stop();
		if (complex_output_checkbox->isChecked() && !is_direct_mode())
			holovibes_.get_pipe()->request_complex_output_stop();
		if (!is_direct_mode())
			global_visibility(true);
		record_but_cancel_visible(true);
	}

	void MainWindow::average_record()
	{
		if (plot_window_)
		{
			plot_window_->stop_drawing();
			plot_window_.reset(nullptr);
			holovibes_.get_pipe()->request_refresh();
		}

		QSpinBox* nb_of_frames_spin_box = findChild<QSpinBox*>("numberOfFramesSpinBox");
		nb_frames_ = nb_of_frames_spin_box->value();
		QLineEdit* output_line_edit = findChild<QLineEdit*>("ROIOutputLineEdit");
		std::string output_path = output_line_edit->text().toUtf8();

		CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
			holovibes_.get_average_queue(),
			output_path,
			nb_frames_,
			this));
		connect(CSV_record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_average_record()));
		CSV_record_thread_->start();

		global_visibility(false);
		record_but_cancel_visible(false);
		average_record_but_cancel_visible(false);
		QPushButton* roi_stop_push_button = findChild<QPushButton*>("ROIStopPushButton");
		roi_stop_push_button->setDisabled(false);
	}

	void MainWindow::finished_average_record()
	{
		CSV_record_thread_.reset(nullptr);
		display_info("ROI record done");

		global_visibility(true);
		record_but_cancel_visible(true);
		average_record_but_cancel_visible(true);
		QPushButton* roi_stop_push_button = findChild<QPushButton*>("ROIStopPushButton");
		roi_stop_push_button->setDisabled(true);
	}

	void MainWindow::browse_batch_input()
	{
		QString filename = QFileDialog::getOpenFileName(this,
			tr("Batch input file"), "C://", tr("All files (*)"));

		QLineEdit* batch_input_line_edit = findChild<QLineEdit*>("batchInputLineEdit");
		batch_input_line_edit->clear();
		batch_input_line_edit->insert(filename);
	}

	void MainWindow::image_batch_record()
	{
		QLineEdit* output_path = findChild<QLineEdit*>("pathLineEdit");

		is_batch_img_ = true;
		is_batch_interrupted_ = false;
		batch_record(std::string(output_path->text().toUtf8()));
	}

	void MainWindow::csv_batch_record()
	{
		if (plot_window_)
		{
			plot_window_->stop_drawing();
			plot_window_.reset(nullptr);
			holovibes_.get_pipe()->request_refresh();
		}

		QLineEdit* output_path = findChild<QLineEdit*>("ROIOutputLineEdit");

		is_batch_img_ = false;
		is_batch_interrupted_ = false;
		batch_record(std::string(output_path->text().toUtf8()));
	}

	void MainWindow::batch_record(const std::string& path)
	{
		file_index_ = 1;
		//struct stat buff;
		QLineEdit* batch_input_line_edit = findChild<QLineEdit*>("batchInputLineEdit");
		QSpinBox * frame_nb_spin_box = findChild<QSpinBox*>("numberOfFramesSpinBox");

		// Getting the path to the input batch file, and the number of frames to record.
		const std::string input_path = batch_input_line_edit->text().toUtf8();
		const unsigned int frame_nb = frame_nb_spin_box->value();

		try
		{
			// Only loading the dll at runtime
			gpib_interface_ = gpib::GpibDLL::load_gpib("gpib.dll", input_path);

			const std::string formatted_path = format_batch_output(path, file_index_);

			global_visibility(false);
			camera_visible(false);

			holovibes::Queue* q;

			if (is_direct_mode())
				q = &holovibes_.get_capture_queue();
			else
				q = &holovibes_.get_output_queue();

			if (gpib_interface_->execute_next_block()) // More blocks to come, use batch_next_block method.
			{
				if (is_batch_img_)
				{
					record_thread_.reset(new ThreadRecorder(*q, formatted_path, frame_nb, this));
					connect(record_thread_.get(),
						SIGNAL(finished()),
						this,
						SLOT(batch_next_record()),
						Qt::UniqueConnection);
					record_thread_->start();
				}
				else
				{
					CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
						holovibes_.get_average_queue(),
						formatted_path,
						frame_nb,
						this));
					connect(CSV_record_thread_.get(),
						SIGNAL(finished()),
						this,
						SLOT(batch_next_record()),
						Qt::UniqueConnection);
					CSV_record_thread_->start();
				}
			}
			else // There was only one block, so no need to record any further.
			{
				if (is_batch_img_)
				{
					record_thread_.reset(new ThreadRecorder(*q, formatted_path, frame_nb, this));
					connect(record_thread_.get(),
						SIGNAL(finished()),
						this,
						SLOT(batch_finished_record()),
						Qt::UniqueConnection);
					record_thread_->start();
				}
				else
				{
					CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
						holovibes_.get_average_queue(),
						formatted_path,
						frame_nb,
						this));
					connect(CSV_record_thread_.get(),
						SIGNAL(finished()),
						this,
						SLOT(batch_finished_record()),
						Qt::UniqueConnection);
					CSV_record_thread_->start();
				}
			}

			// Update the index to concatenate to the name of the next created file.
			++file_index_;
		}
		catch (const std::exception& e)
		{
			display_error(e.what());
			batch_finished_record(false);
		}
	}

	void MainWindow::batch_next_record()
	{
		if (is_batch_interrupted_)
		{
			batch_finished_record(false);
			return;
		}

		record_thread_.reset(nullptr);

		QSpinBox * frame_nb_spin_box = findChild<QSpinBox*>("numberOfFramesSpinBox");
		std::string path;

		if (is_batch_img_)
			path = findChild<QLineEdit*>("pathLineEdit")->text().toUtf8();
		else
			path = findChild<QLineEdit*>("ROIOutputLineEdit")->text().toUtf8();

		holovibes::Queue* q;
		if (is_direct_mode())
			q = &holovibes_.get_capture_queue();
		else
			q = &holovibes_.get_output_queue();

		std::string output_filename = format_batch_output(path, file_index_);
		const unsigned int frame_nb = frame_nb_spin_box->value();
		if (is_batch_img_)
		{
			try
			{
				if (gpib_interface_->execute_next_block())
				{
					record_thread_.reset(new ThreadRecorder(*q, output_filename, frame_nb, this));
					connect(record_thread_.get(),
						SIGNAL(finished()),
						this,
						SLOT(batch_next_record()), Qt::UniqueConnection);
					record_thread_->start();
				}
				else
				{
					batch_finished_record(true);
				}
			}
			catch (const gpib::GpibInstrError& e)
			{
				display_error(e.what());
				batch_finished_record(false);
			}
		}
		else
		{
			try
			{
				if (gpib_interface_->execute_next_block())
				{
					CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
						holovibes_.get_average_queue(),
						output_filename,
						frame_nb,
						this));
					connect(CSV_record_thread_.get(),
						SIGNAL(finished()),
						this,
						SLOT(batch_next_record()), Qt::UniqueConnection);
					CSV_record_thread_->start();
				}
				else
					batch_finished_record(true);
			}
			catch (const gpib::GpibInstrError& e)
			{
				display_error(e.what());
				batch_finished_record(false);
			}
		}

		// Update the index to concatenate to the name of the next created file.
		++file_index_;
	}

	void MainWindow::batch_finished_record()
	{
		batch_finished_record(true);
	}

	void MainWindow::batch_finished_record(bool no_error)
	{
		record_thread_.reset(nullptr);
		CSV_record_thread_.reset(nullptr);
		gpib_interface_.reset();

		file_index_ = 1;
		if (!is_direct_mode())
			global_visibility(true);
		camera_visible(true);
		if (no_error)
			display_info("Batch record done");

		if (plot_window_)
		{
			plot_window_->stop_drawing();
			holovibes_.get_pipe()->request_average(&holovibes_.get_average_queue());
			plot_window_->start_drawing();
		}
	}

	void MainWindow::stop_image_record()
	{
		if (record_thread_)
		{
			record_thread_->stop();
			is_batch_interrupted_ = true;
		}
	}

	void MainWindow::stop_csv_record()
	{
		if (is_enabled_average_)
		{
			if (CSV_record_thread_)
			{
				CSV_record_thread_->stop();
				is_batch_interrupted_ = true;
			}
		}
	}

	void MainWindow::set_float_visible(bool value)
	{
		QCheckBox* complex_checkbox = findChild<QCheckBox*>("RecordComplexOutputCheckBox");
		if (complex_checkbox->isChecked() && value == true)
			complex_checkbox->setChecked(false);
	}

	void MainWindow::set_complex_visible(bool value)
	{
		QCheckBox* float_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");
		if (float_checkbox->isChecked() && value == true)
			float_checkbox->setChecked(false);
	}

	void MainWindow::import_browse_file()
	{
		QString filename = QFileDialog::getOpenFileName(this,
			tr("import file"), "C://", tr("All files (*)"));

		QLineEdit* import_line_edit = findChild<QLineEdit*>("ImportPathLineEdit");
		import_line_edit->clear();
		import_line_edit->insert(filename);
	}

	void MainWindow::import_file_stop(void)
	{
		close_critical_compute();
		camera_none();
	}

	void MainWindow::import_file()
	{
		close_critical_compute();
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
		QLineEdit* import_line_edit = findChild<QLineEdit*>("ImportPathLineEdit");
		QSpinBox* width_spinbox = findChild<QSpinBox*>("ImportWidthSpinBox");
		QSpinBox* height_spinbox = findChild<QSpinBox*>("ImportHeightSpinBox");
		QSpinBox* fps_spinbox = findChild<QSpinBox*>("ImportFpsSpinBox");
		QSpinBox* start_spinbox = findChild<QSpinBox*>("ImportStartSpinBox");
		QSpinBox* end_spinbox = findChild<QSpinBox*>("ImportEndSpinBox");
		QComboBox* depth_spinbox = findChild<QComboBox*>("ImportDepthModeComboBox");
		//QCheckBox* squared_checkbox = findChild<QCheckBox*>("ImportSquaredCheckBox");
		QComboBox* big_endian_checkbox = findChild<QComboBox*>("ImportEndianModeComboBox");
		QCheckBox* cine = findChild<QCheckBox*>("CineFileCheckBox");
		cd.stft_steps.exchange(std::ceil(static_cast<float>(fps_spinbox->value()) / 20.0f));
		int	depth_multi = 1;
		std::string file_src = import_line_edit->text().toUtf8();

		if (file_src == "")
			return;
		try
		{
			if (cine->isChecked() == true)
				seek_cine_header_data(file_src, holovibes_);
		}
		catch (std::exception& e)
		{
			display_error(e.what());
			return;
		}

		if (depth_spinbox->currentIndex() == 1)
			depth_multi = 2;
		else if (depth_spinbox->currentIndex() == 2)
			depth_multi = 4;
		else if (depth_spinbox->currentIndex() == 3)
			depth_multi = 8;
		camera::FrameDescriptor frame_desc = {
			width_spinbox->value(),
			height_spinbox->value(),
			depth_multi,
			cd.import_pixel_size,
			(big_endian_checkbox->currentText() == QString("Big Endian") ?
				camera::endianness::BIG_ENDIAN : camera::endianness::LITTLE_ENDIAN) };
		camera_visible(false);
		record_visible(false);
		global_visibility(false);
		gl_window_.reset(nullptr);
		holovibes_.dispose_compute();
		holovibes_.dispose_capture();
		try
		{
			holovibes_.init_import_mode(
				file_src,
				frame_desc,
				true,
				fps_spinbox->value(),
				start_spinbox->value(),
				end_spinbox->value(),
				global::global_config.input_queue_max_size,
				holovibes_);
		}
		catch (std::exception& e)
		{
			display_error(e.what());
			camera_visible(false);
			record_visible(false);
			global_visibility(false);
			gl_window_.reset(nullptr);
			holovibes_.dispose_compute();
			holovibes_.dispose_capture();
			return;
		}
		camera_visible(true);
		record_visible(true);
		set_image_mode();

		// Changing the gui
		QLineEdit* boundary = findChild<QLineEdit*>("boundary");
		boundary->clear();
		boundary->insert(QString::number(holovibes_.get_boundary()));
		if (depth_spinbox->currentText() == QString("16") && cine->isChecked() == false)
			big_endian_checkbox->setEnabled(true);
		QAction* settings = findChild<QAction*>("actionSettings");
		settings->setEnabled(false);
		if (holovibes_.get_tcapture()->stop_requested_)
		{
			camera_visible(false);
			record_visible(false);
			global_visibility(false);
			gl_window_.reset(nullptr);
			holovibes_.dispose_compute();
			holovibes_.dispose_capture();
		}
	}

	void MainWindow::import_start_spinbox_update()
	{
		QSpinBox* start_spinbox = findChild<QSpinBox*>("ImportStartSpinBox");
		QSpinBox* end_spinbox = findChild<QSpinBox*>("ImportEndSpinBox");

		if (start_spinbox->value() > end_spinbox->value())
			end_spinbox->setValue(start_spinbox->value());
	}

	void MainWindow::import_end_spinbox_update()
	{
		QSpinBox* start_spinbox = findChild<QSpinBox*>("ImportStartSpinBox");
		QSpinBox* end_spinbox = findChild<QSpinBox*>("ImportEndSpinBox");

		if (end_spinbox->value() < start_spinbox->value())
			start_spinbox->setValue(end_spinbox->value());
	}

	void MainWindow::closeEvent(QCloseEvent* event)
	{
		// Avoiding "unused variable" warning.
		static_cast<void>(event);
		save_ini("holovibes.ini");

		if (gl_window_)
			gl_window_->close();

		if (plot_window_)
			plot_window_->close();
	}

	void MainWindow::global_visibility(const bool value)
	{
		GroupBox* view = findChild<GroupBox*>("View");
		view->setDisabled(!value);

		GroupBox* special = findChild<GroupBox*>("Vibrometry");
		special->setDisabled(!value);

		QPushButton* takeref = findChild<QPushButton*>("takerefPushButton");
		takeref->setDisabled(!value);

		QPushButton* sliding = findChild<QPushButton*>("slindingrefPushButton");
		sliding->setDisabled(!value);

		QPushButton* filter2D = findChild<QPushButton*>("Filter2DPushButton");
		filter2D->setDisabled(!value);

		QLabel* phase_number_label = findChild<QLabel*>("PhaseNumberLabel");
		phase_number_label->setDisabled(!value);

		QSpinBox* phase_nb = findChild<QSpinBox*>("phaseNumberSpinBox");
		phase_nb->setDisabled(!value);

		QLabel* p_label = findChild<QLabel*>("pLabel");
		p_label->setDisabled(!value);

		QSpinBox* p = findChild<QSpinBox*>("pSpinBox");
		p->setDisabled(!value);

		QLabel* wavelength_label = findChild<QLabel*>("wavelengthLabel");
		wavelength_label->setDisabled(!value);

		QDoubleSpinBox* wavelength = findChild<QDoubleSpinBox*>("wavelengthSpinBox");
		wavelength->setDisabled(!value);

		QLabel* z_label = findChild<QLabel*>("zLabel");
		z_label->setDisabled(!value);

		QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
		z->setDisabled(!value);

		QLabel* z_step_label = findChild<QLabel*>("zStepLabel");
		z_step_label->setDisabled(!value);

		QDoubleSpinBox* z_step = findChild<QDoubleSpinBox*>("zStepDoubleSpinBox");
		z_step->setDisabled(!value);

		QLabel* algorithm_label = findChild<QLabel*>("algorithmLabel");
		algorithm_label->setDisabled(!value);

		QComboBox* algorithm = findChild<QComboBox*>("algorithmComboBox");
		algorithm->setDisabled(!value);

		QCheckBox* float_output_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");
		float_output_checkbox->setDisabled(!value);

		QCheckBox* complex_output_checkbox = findChild<QCheckBox*>("RecordComplexOutputCheckBox");
		complex_output_checkbox->setDisabled(!value);

		QLineEdit* boundary = findChild<QLineEdit*>("boundary");
		boundary->setDisabled(!value);

		QCheckBox* stft_button = findChild<QCheckBox*>("STFTCheckBox");
		stft_button->setDisabled(!value);

		QCheckBox* cine = findChild<QCheckBox*>("CineFileCheckBox");
		QComboBox* depth_spinbox = findChild<QComboBox*>("ImportDepthModeComboBox");
		QComboBox* big_endian_checkbox = findChild<QComboBox*>("ImportEndianModeComboBox");
		if (cine->isChecked() == false)
		{
			QDoubleSpinBox* import_pixel_size = findChild<QDoubleSpinBox*>("ImportPixelSizeDoubleSpinBox");
			import_pixel_size->setEnabled(true);
			if (depth_spinbox->currentText() == QString("16"))
				big_endian_checkbox->setEnabled(true);
		}
		else if (cine->isChecked() == true)
		{
			if (depth_spinbox->currentText() == QString("16"))
				big_endian_checkbox->setEnabled(false);
		}
	}

	void MainWindow::phase_num_visible(const bool value)
	{
		QSpinBox* phase_num = findChild<QSpinBox*>("phaseNumberSpinBox");
		phase_num->setDisabled(!value);
	}

	void MainWindow::demodulation_visibility(const bool value)
	{
		QLabel* wavelength_label = findChild<QLabel*>("wavelengthLabel");
		wavelength_label->setDisabled(!value);

		QDoubleSpinBox* wavelength = findChild<QDoubleSpinBox*>("wavelengthSpinBox");
		wavelength->setDisabled(!value);

		QLabel* z_label = findChild<QLabel*>("zLabel");
		z_label->setDisabled(!value);

		QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
		z->setDisabled(!value);

		QLabel* z_step_label = findChild<QLabel*>("zStepLabel");
		z_step_label->setDisabled(!value);

		QDoubleSpinBox* z_step = findChild<QDoubleSpinBox*>("zStepDoubleSpinBox");
		z_step->setDisabled(!value);

		QLabel* algorithm_label = findChild<QLabel*>("algorithmLabel");
		algorithm_label->setDisabled(!value);

		QComboBox* algorithm = findChild<QComboBox*>("algorithmComboBox");
		algorithm->setDisabled(!value);
	}

	void MainWindow::camera_visible(const bool value)
	{
		is_enabled_camera_ = value;
		QRadioButton* direct = findChild<QRadioButton*>("directImageRadioButton");
		direct->setEnabled(value);
		QRadioButton* holo = findChild<QRadioButton*>("hologramRadioButton");
		holo->setEnabled(value);
	}

	void MainWindow::contrast_visible(const bool value)
	{
		QLabel* min_label = findChild<QLabel*>("minLabel");
		QLabel* max_label = findChild<QLabel*>("maxLabel");
		QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
		QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");

		min_label->setDisabled(!value);
		max_label->setDisabled(!value);
		contrast_min->setDisabled(!value);
		contrast_max->setDisabled(!value);
	}

	void MainWindow::record_visible(const bool value)
	{
		gui::GroupBox* image_rendering = findChild<gui::GroupBox*>("Record");
		image_rendering->setDisabled(!value);
	}

	void MainWindow::record_but_cancel_visible(const bool value)
	{
		QLabel* nb_of_frames_label = findChild<QLabel*>("numberOfFramesLabel");
		nb_of_frames_label->setDisabled(!value);
		QSpinBox* nb_of_frames_spinbox = findChild<QSpinBox*>("numberOfFramesSpinBox");
		nb_of_frames_spinbox->setDisabled(!value);
		QToolButton* browse_button = findChild<QToolButton*>("ImageOutputBrowsePushButton");
		browse_button->setDisabled(!value);
		QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
		path_line_edit->setDisabled(!value);
		QPushButton* record_button = findChild<QPushButton*>("recPushButton");
		record_button->setDisabled(!value);
	}

	void MainWindow::image_ratio_visible(const bool value)
	{
		QLabel* p_label_vibro = findChild<QLabel*>("pLabelVibro");
		p_label_vibro->setDisabled(!value);
		QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
		p_vibro->setDisabled(!value);
		QLabel* q_label_vibro = findChild<QLabel*>("qLabelVibro");
		q_label_vibro->setDisabled(!value);
		QSpinBox* q_vibro = findChild<QSpinBox*>("qSpinBoxVibro");
		q_vibro->setDisabled(!value);
	}

	void MainWindow::average_visible(const bool value)
	{
		QLabel* roi_file_label = findChild<QLabel*>("ROIFileLabel");
		roi_file_label->setDisabled(!value);
		QToolButton* roi_browse_button = findChild<QToolButton*>("ROIFileBrowseToolButton");
		roi_browse_button->setDisabled(!value);
		QLineEdit* roi_file_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
		roi_file_line_edit->setDisabled(!value);
		QPushButton* save_roi_button = findChild<QPushButton*>("saveROIPushButton");
		save_roi_button->setDisabled(!value);
		QPushButton* load_roi_button = findChild<QPushButton*>("loadROIPushButton");
		load_roi_button->setDisabled(!value);
	}

	void MainWindow::average_record_but_cancel_visible(const bool value)
	{
		QLabel* roi_output_file_label = findChild<QLabel*>("ROIOuputLabel");
		roi_output_file_label->setDisabled(!value);
		QToolButton* roi_output_push_button = findChild<QToolButton*>("ROIOutputToolButton");
		roi_output_push_button->setDisabled(!value);
		QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("ROIOutputLineEdit");
		roi_output_line_edit->setDisabled(!value);
		QPushButton* roi_push_button = findChild<QPushButton*>("ROIPushButton");
		roi_push_button->setDisabled(!value);
	}

	void MainWindow::change_camera(const holovibes::Holovibes::camera_type camera_type)
	{
		close_critical_compute();
		/*auto manager = gui::InfoManager::get_manager();
		manager->remove_info_safe("Input Fps");*/
		gui::InfoManager::get_manager()->remove_info_safe("Input Fps");
		if (camera_type != holovibes::Holovibes::NONE)
		{
			try
			{
				camera_visible(false);
				record_visible(false);
				global_visibility(false);
				gl_window_.reset(nullptr);
				holovibes_.dispose_compute();
				holovibes_.dispose_capture();
				holovibes_.init_capture(camera_type);
				camera_visible(true);
				record_visible(true);
				set_image_mode();
				camera_type_ = camera_type;

				// Changing the gui
				holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

				QDoubleSpinBox* import_pixel_size = findChild<QDoubleSpinBox*>("ImportPixelSizeDoubleSpinBox");
				import_pixel_size->setValue(holovibes_.get_cam_frame_desc().pixel_size);
				cd.import_pixel_size.exchange(holovibes_.get_cam_frame_desc().pixel_size);

				QLineEdit* boundary = findChild<QLineEdit*>("boundary");
				boundary->clear();
				boundary->insert(QString::number(holovibes_.get_boundary()));

				QAction* settings = findChild<QAction*>("actionSettings");
				settings->setEnabled(true);
			}
			catch (camera::CameraException& e)
			{
				display_error("[CAMERA]" + std::string(e.what()));
			}
			catch (std::exception& e)
			{
				display_error(e.what());
			}
		}
	}

	void MainWindow::display_error(const std::string msg)
	{
		QMessageBox msg_box;
		msg_box.setText(QString::fromLatin1(msg.c_str()));
		msg_box.setIcon(QMessageBox::Critical);
		msg_box.exec();
	}

	void MainWindow::display_info(const std::string msg)
	{
		QMessageBox msg_box;
		msg_box.setText(QString::fromLatin1(msg.c_str()));
		msg_box.setIcon(QMessageBox::Information);
		msg_box.exec();
	}

	void MainWindow::open_file(const std::string& path)
	{
		QDesktopServices::openUrl(QUrl::fromLocalFile(QString(path.c_str())));
	}

	void MainWindow::load_ini(const std::string& path)
	{
		boost::property_tree::ptree ptree;
		gui::GroupBox *image_rendering_group_box = findChild<gui::GroupBox*>("ImageRendering");
		gui::GroupBox *view_group_box = findChild<gui::GroupBox*>("View");
		gui::GroupBox *special_group_box = findChild<gui::GroupBox*>("Vibrometry");
		gui::GroupBox *record_group_box = findChild<gui::GroupBox*>("Record");
		gui::GroupBox *import_group_box = findChild<gui::GroupBox*>("Import");
		gui::GroupBox *info_group_box = findChild<gui::GroupBox*>("Info");

		QAction*      image_rendering_action = findChild<QAction*>("actionImage_rendering");
		QAction*      view_action = findChild<QAction*>("actionView");
		QAction*      special_action = findChild<QAction*>("actionSpecial");
		QAction*      record_action = findChild<QAction*>("actionRecord");
		QAction*      import_action = findChild<QAction*>("actionImport");
		QAction*      info_action = findChild<QAction*>("actionInfo");

		try
		{
			boost::property_tree::ini_parser::read_ini(path, ptree);
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}

		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

		if (!ptree.empty())
		{
			holovibes::Config& config = global::global_config;
			// Config
			config.input_queue_max_size = ptree.get<int>("config.input_buffer_size", config.input_queue_max_size);
			config.output_queue_max_size = ptree.get<int>("config.output_buffer_size", config.output_queue_max_size);
			config.float_queue_max_size = ptree.get<int>("config.float_buffer_size", config.float_queue_max_size);
			config.frame_timeout = ptree.get<int>("config.frame_timeout", config.frame_timeout);
			config.flush_on_refresh = ptree.get<int>("config.flush_on_refresh", config.flush_on_refresh);
			config.reader_buf_max_size = ptree.get<int>("config.input_file_buffer_size", config.reader_buf_max_size);
			cd.special_buffer_size.exchange(ptree.get<int>("config.convolution_buffer_size", cd.special_buffer_size));
			cd.stft_level = ptree.get<unsigned int>("config.stft_buffer_size", cd.stft_level);
			cd.ref_diff_level = ptree.get<unsigned int>("config.reference_buffer_size", cd.ref_diff_level);
			cd.img_acc_level = ptree.get<unsigned int>("config.accumulation_buffer_size", cd.img_acc_level);

			// Camera type
			const int camera_type = ptree.get<int>("image_rendering.camera", 0);
			change_camera((holovibes::Holovibes::camera_type)camera_type);

			// Image rendering
			image_rendering_action->setChecked(!ptree.get<bool>("image_rendering.hidden", false));
			image_rendering_group_box->setHidden(ptree.get<bool>("image_rendering.hidden", false));

			const unsigned short p_nsample = ptree.get<unsigned short>("image_rendering.phase_number", cd.nsamples);
			if (p_nsample < 1)
				cd.nsamples.exchange(1);
			else if (p_nsample > config.input_queue_max_size)
				cd.nsamples.exchange(config.input_queue_max_size);
			else
				cd.nsamples.exchange(p_nsample);

			const unsigned short p_index = ptree.get<unsigned short>("image_rendering.p_index", cd.pindex);
			if (p_index >= 0 && p_index < cd.nsamples)
				cd.pindex.exchange(p_index);

			cd.lambda = ptree.get<float>("image_rendering.lambda", cd.lambda);

			cd.zdistance = ptree.get<float>("image_rendering.z_distance", cd.zdistance);

			const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
			if (z_step > 0.0f)
				z_step_ = z_step;

			cd.algorithm = static_cast<holovibes::ComputeDescriptor::fft_algorithm>(
				ptree.get<int>("image_rendering.algorithm", cd.algorithm));

			// View
			view_action->setChecked(!ptree.get<bool>("view.hidden", false));
			view_group_box->setHidden(ptree.get<bool>("view.hidden", false));

			cd.view_mode = static_cast<holovibes::ComputeDescriptor::complex_view_mode>(
				ptree.get<int>("view.view_mode", cd.view_mode));

			cd.log_scale_enabled.exchange(
				ptree.get<bool>("view.log_scale_enabled", cd.log_scale_enabled));

			cd.shift_corners_enabled.exchange(
				ptree.get<bool>("view.shift_corners_enabled", cd.shift_corners_enabled));

			cd.contrast_enabled.exchange(
				ptree.get<bool>("view.contrast_enabled", cd.contrast_enabled));

			cd.contrast_min = ptree.get<float>("view.contrast_min", cd.contrast_min);

			cd.contrast_max = ptree.get<float>("view.contrast_max", cd.contrast_max);

			cd.img_acc_enabled = ptree.get<bool>("view.accumulation_enabled", cd.img_acc_enabled);

			// Post Processing
			special_action->setChecked(!ptree.get<bool>("post_processing.hidden", false));
			special_group_box->setHidden(ptree.get<bool>("post_processing.hidden", false));
			/* cd.vibrometry_enabled.exchange(
			   ptree.get<bool>("post_processing.image_ratio_enabled", cd.vibrometry_enabled));*/
			cd.vibrometry_q.exchange(
				ptree.get<int>("post_processing.image_ratio_q", cd.vibrometry_q));
			is_enabled_average_ = ptree.get<bool>("post_processing.average_enabled", is_enabled_average_);


			// Record
			record_action->setChecked(!ptree.get<bool>("record.hidden", false));
			record_group_box->setHidden(ptree.get<bool>("record.hidden", false));

			// Import
			import_action->setChecked(!ptree.get<bool>("import.hidden", false));
			import_group_box->setHidden(ptree.get<bool>("import.hidden", false));
			config.import_pixel_size = ptree.get<float>("import.pixel_size", config.import_pixel_size);
			cd.import_pixel_size = config.import_pixel_size;

			// Info
			info_action->setChecked(!ptree.get<bool>("info.hidden", false));
			info_group_box->setHidden(ptree.get<bool>("info.hidden", false));
			theme_index_ = ptree.get<int>("info.theme_type", theme_index_);

			// Autofocus
			cd.autofocus_size = ptree.get<int>("autofocus.size", cd.autofocus_size);
			cd.autofocus_z_min = ptree.get<float>("autofocus.z_min", cd.autofocus_z_min);
			cd.autofocus_z_max = ptree.get<float>("autofocus.z_max", cd.autofocus_z_max);
			cd.autofocus_z_div = ptree.get<unsigned int>("autofocus.steps", cd.autofocus_z_div);
			cd.autofocus_z_iter = ptree.get<unsigned int>("autofocus.loops", cd.autofocus_z_iter);

			//flowgraphy
			unsigned int flowgraphy_level = ptree.get<unsigned int>("flowgraphy.level", cd.flowgraphy_level);
			if (flowgraphy_level % 2 == 0)
				flowgraphy_level++;
			cd.flowgraphy_level.exchange(flowgraphy_level);
			cd.flowgraphy_enabled = ptree.get<bool>("flowgraphy.enable", cd.flowgraphy_enabled);

			// Reset button
			config.set_cuda_device = ptree.get<bool>("reset.set_cuda_device", config.set_cuda_device);
			config.auto_device_number = ptree.get<bool>("reset.auto_device_number", config.auto_device_number);
			config.device_number = ptree.get<int>("reset.device_number", config.device_number);

		}
	}

	void MainWindow::save_ini(const std::string& path)
	{
		boost::property_tree::ptree ptree;
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
		if (cd.stft_enabled)
		{
			unsigned int tmp = cd.nsamples.load();
			cd.nsamples.exchange(cd.stft_level.load());
			cd.stft_level.exchange(tmp);
		}
		gui::GroupBox *image_rendering_group_box = findChild<gui::GroupBox*>("ImageRendering");
		gui::GroupBox *view_group_box = findChild<gui::GroupBox*>("View");
		gui::GroupBox *special_group_box = findChild<gui::GroupBox*>("Vibrometry");
		gui::GroupBox *record_group_box = findChild<gui::GroupBox*>("Record");
		gui::GroupBox *import_group_box = findChild<gui::GroupBox*>("Import");
		gui::GroupBox *info_group_box = findChild<gui::GroupBox*>("Info");
		holovibes::Config& config = global::global_config;

		// Config
		ptree.put("config.input_buffer_size", config.input_queue_max_size);
		ptree.put("config.output_buffer_size", config.output_queue_max_size);
		ptree.put("config.float_buffer_size", config.float_queue_max_size);
		ptree.put("config.input_file_buffer_size", config.reader_buf_max_size);
		ptree.put("config.stft_buffer_size", cd.stft_level);
		ptree.put("config.reference_buffer_size", cd.ref_diff_level);
		ptree.put("config.accumulation_buffer_size", cd.img_acc_level);
		ptree.put("config.convolution_buffer_size", cd.special_buffer_size);
		ptree.put("config.frame_timeout", config.frame_timeout);
		ptree.put<bool>("config.flush_on_refresh", config.flush_on_refresh);

		// Image rendering
		ptree.put<bool>("image_rendering.hidden", image_rendering_group_box->isHidden());
		ptree.put("image_rendering.camera", camera_type_);
		ptree.put("image_rendering.phase_number", cd.nsamples);
		ptree.put("image_rendering.p_index", cd.pindex);
		ptree.put("image_rendering.lambda", cd.lambda);
		ptree.put("image_rendering.z_distance", cd.zdistance);
		ptree.put("image_rendering.z_step", z_step_);
		ptree.put("image_rendering.algorithm", cd.algorithm);

		// View
		ptree.put<bool>("view.hidden", view_group_box->isHidden());
		ptree.put("view.view_mode", cd.view_mode);
		ptree.put<bool>("view.log_scale_enabled", cd.log_scale_enabled);
		ptree.put<bool>("view.shift_corners_enabled", cd.shift_corners_enabled);
		ptree.put<bool>("view.contrast_enabled", cd.contrast_enabled);
		ptree.put("view.contrast_min", cd.contrast_min);
		ptree.put("view.contrast_max", cd.contrast_max);
		ptree.put<bool>("view.accumulation_enabled", cd.img_acc_enabled);

		// Post-processing
		ptree.put<bool>("post_processing.hidden", special_group_box->isHidden());
		//ptree.put("post_processing.image_ratio_enabled", cd.vibrometry_enabled);
		ptree.put("post_processing.image_ratio_q", cd.vibrometry_q);
		ptree.put<bool>("post_processing.average_enabled", is_enabled_average_);

		// Record
		ptree.put<bool>("record.hidden", record_group_box->isHidden());

		// Import
		ptree.put<bool>("import.hidden", import_group_box->isHidden());
		ptree.put("import.pixel_size", cd.import_pixel_size);

		// Info
		ptree.put<bool>("info.hidden", info_group_box->isHidden());
		ptree.put("info.theme_type", theme_index_);

		// Autofocus
		ptree.put("autofocus.size", cd.autofocus_size);
		ptree.put("autofocus.z_min", cd.autofocus_z_min);
		ptree.put("autofocus.z_max", cd.autofocus_z_max);
		ptree.put("autofocus.steps", cd.autofocus_z_div);
		ptree.put("autofocus.loops", cd.autofocus_z_iter);

		//flowgraphy
		ptree.put("flowgraphy.level", cd.flowgraphy_level);
		ptree.put<bool>("flowgraphy.enable", cd.flowgraphy_enabled);

		//Reset
		ptree.put<bool>("reset.set_cuda_device", config.set_cuda_device);
		ptree.put<bool>("reset.auto_device_number", config.auto_device_number);
		ptree.put("reset.device_number", config.device_number);


		boost::property_tree::write_ini(holovibes_.get_launch_path() + "/" + path, ptree);
	}

	void MainWindow::split_string(const std::string& str, const char delim, std::vector<std::string>& elts)
	{
		std::stringstream ss(str);
		std::string item;

		while (std::getline(ss, item, delim))
			elts.push_back(item);
	}

	std::string MainWindow::format_batch_output(const std::string& path, const unsigned int index)
	{
		std::string file_index;
		std::ostringstream convert;
		convert << std::setw(6) << std::setfill('0') << index;
		file_index = convert.str();

		std::vector<std::string> path_tokens;
		split_string(path, '.', path_tokens);

		return path_tokens[0] + "_" + file_index + "." + path_tokens[1];
	}

	void MainWindow::hide_endianess()
	{
		QComboBox* depth_cbox = findChild<QComboBox*>("ImportDepthModeComboBox");
		QString curr_value = depth_cbox->currentText();
		QComboBox* imp_cbox = findChild<QComboBox*>("ImportEndianModeComboBox");

		// Changing the endianess when depth = 8 makes no sense
		imp_cbox->setEnabled(curr_value == "16");
	}

	void MainWindow::set_import_cine_file(bool value)
	{
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
		QCheckBox* cine = findChild<QCheckBox*>("CineFileCheckBox");
		QSpinBox* width_spinbox = findChild<QSpinBox*>("ImportWidthSpinBox");
		QSpinBox* height_spinbox = findChild<QSpinBox*>("ImportHeightSpinBox");
		QComboBox* depth_spinbox = findChild<QComboBox*>("ImportDepthModeComboBox");
		QComboBox* big_endian_checkbox = findChild<QComboBox*>("ImportEndianModeComboBox");
		QDoubleSpinBox* import_pixel_size = findChild<QDoubleSpinBox*>("ImportPixelSizeDoubleSpinBox");

		cd.is_cine_file.exchange(value);
		cine->setChecked(value);
		width_spinbox->setEnabled(!value);
		height_spinbox->setEnabled(!value);
		depth_spinbox->setEnabled(!value);
		import_pixel_size->setEnabled(!value);
		if (depth_spinbox->currentText() == QString("16"))
			big_endian_checkbox->setEnabled(!value);
	}

	void MainWindow::seek_cine_header_data(std::string &file_src_, holovibes::Holovibes& holovibes_)
	{
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
		QSpinBox*		width_spinbox = findChild<QSpinBox*>("ImportWidthSpinBox");
		QSpinBox*		height_spinbox = findChild<QSpinBox*>("ImportHeightSpinBox");
		QComboBox*	depth_spinbox = findChild<QComboBox*>("ImportDepthModeComboBox");
		QComboBox*	big_endian_checkbox = findChild<QComboBox*>("ImportEndianModeComboBox");
		QDoubleSpinBox* import_pixel_size = findChild<QDoubleSpinBox*>("ImportPixelSizeDoubleSpinBox");
		int			read_width = 0;
		int			read_height = 0;
		unsigned short int read_depth = 0;
		unsigned int	read_pixelpermeter_x = 0;
		FILE*			file = nullptr;
		fpos_t		pos = 0;
		size_t		length = 0;
		unsigned int	offset_to_ptr = 0;
		char			buffer[44];
		double		pixel_size = 0;

		try
		{
			/*Opening file and checking if it exists*/
			fopen_s(&file, file_src_.c_str(), "rb");
			if (!file)
				throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
			std::fsetpos(file, &pos);
			/*Reading the whole cine file header*/
			if ((length = std::fread(buffer, 1, 44, file)) = !44)
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
			/*Checking if the file is actually a .cine file*/
			if (std::strstr(buffer, "CI") == NULL)
				throw std::runtime_error("[READER] file " + file_src_ + " is not a valid .cine file");
			/*Reading OffImageHeader for offset to BITMAPINFOHEADER*/
			std::memcpy(&offset_to_ptr, (buffer + 24), sizeof(int));
			/*Reading value biWidth*/
			pos = offset_to_ptr + 4;
			std::fsetpos(file, &pos);
			if ((length = std::fread(&read_width, 1, sizeof(int), file)) = !sizeof(int))
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
			/*Reading value biHeigth*/
			pos = offset_to_ptr + 8;
			std::fsetpos(file, &pos);
			if ((length = std::fread(&read_height, 1, sizeof(int), file)) = !sizeof(int))
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
			/*Reading value biBitCount*/
			pos = offset_to_ptr + 14;
			std::fsetpos(file, &pos);
			if ((length = std::fread(&read_depth, 1, sizeof(short int), file)) = !sizeof(short int))
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
			/*Reading value biXpelsPerMetter*/
			pos = offset_to_ptr + 24;
			std::fsetpos(file, &pos);
			if ((length = std::fread(&read_pixelpermeter_x, 1, sizeof(int), file)) = !sizeof(int))
				throw std::runtime_error("[READER] unable to read file: " + file_src_);

			/*Setting value in Qt interface*/
			if (read_depth == 8)
				depth_spinbox->setCurrentIndex(0);
			else
				depth_spinbox->setCurrentIndex(1);
			width_spinbox->setValue(read_width);
			if (read_height < 0)
				read_height = -read_height;
			height_spinbox->setValue(read_height);
			pixel_size = (1 / (double)read_pixelpermeter_x) * 1000000;
			cd.import_pixel_size = pixel_size;
			import_pixel_size->setValue(cd.import_pixel_size.load());
			big_endian_checkbox->setCurrentText("Little Endian");

			/*Unused fonction ready to read framerate dans exposure*/
			//holovibes::get_framerate_cinefile(file, file_src_);
			//holovibes::get_exposure_cinefile(file, file_src_);
		}
		catch (std::runtime_error& e)
		{
			std::cout << e.what() << std::endl;
			throw std::runtime_error(e.what());
		}
	}

	void MainWindow::close_critical_compute()
	{
		holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
		if (cd.stft_enabled)
		{
			QCheckBox* stft_button = findChild<QCheckBox*>("STFTCheckBox");
			stft_button->setChecked(false);
		}
		if (cd.ref_diff_enabled || cd.ref_sliding_enabled)
			cancel_take_reference();
		if (cd.filter_2d_enabled)
			cancel_filter2D();
	}

	void MainWindow::write_ini()
	{
		close_critical_compute();
		save_ini("holovibes.ini");
		notify();
	}

	void MainWindow::reload_ini()
	{
		close_critical_compute();
		load_ini("holovibes.ini");
		notify();
	}

	void MainWindow::set_classic()
	{
		theme_index_ = 0;
		qApp->setPalette(this->style()->standardPalette());
		qApp->setStyle(QStyleFactory::create("WindowsVista"));
		qApp->setStyleSheet("");
	}

	void MainWindow::set_night()
	{
		theme_index_ = 1;
		qApp->setStyle(QStyleFactory::create("Fusion"));

		QPalette darkPalette;
		darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
		darkPalette.setColor(QPalette::WindowText, Qt::white);
		darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
		darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
		darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
		darkPalette.setColor(QPalette::ToolTipText, Qt::white);
		darkPalette.setColor(QPalette::Text, Qt::white);
		darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
		darkPalette.setColor(QPalette::ButtonText, Qt::white);
		darkPalette.setColor(QPalette::BrightText, Qt::red);
		darkPalette.setColor(QPalette::Disabled, QPalette::Text, Qt::darkGray);
		darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, Qt::darkGray);
		darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, Qt::darkGray);
		darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
		darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
		darkPalette.setColor(QPalette::HighlightedText, Qt::black);

		qApp->setPalette(darkPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}

	void MainWindow::stft_view(bool b)
	{
		QCheckBox*	p = findChild<QCheckBox*>("STFTCheckBox");
		holovibes::ComputeDescriptor&	cd = holovibes_.get_compute_desc();
		if (b)
		{
			p->setEnabled(false);
			// launch stft_view windows
			notify();
			holovibes_.get_pipe()->create_stft_slice_queue();
<<<<<<< 486a9e8aabedea0898758ee00c46867dcb5b2f64
			gl_win_stft_0.reset(new GuiGLWindow(
				QPoint(520, 0), 512, 512, holovibes_, holovibes_.get_pipe()->get_stft_slice_queue(0)));
			gl_win_stft_1.reset(new GuiGLWindow(
				QPoint(0, 545), 512, 512, holovibes_, holovibes_.get_pipe()->get_stft_slice_queue(1)));
=======
			
			gl_win_stft_0.reset(new GuiGLWindow(
				QPoint(512, 0), 512, 512, holovibes_, holovibes_.get_pipe()->get_stft_slice_queue(), GuiGLWindow::window_kind::SLICE_XZ));
>>>>>>> Update : slicing is done in a brand new window & bug fix when deleting window

			cd.stft_view_enabled.exchange(true);
		}
		else
		{
			// delete stft_view windows
			cd.stft_view_enabled.exchange(false);
			gl_win_stft_1.reset(nullptr);
			gl_win_stft_0.reset(nullptr);
			holovibes_.get_pipe()->delete_stft_slice_queue();
			// -------------------
			p->setEnabled(true);
		}
	}
}