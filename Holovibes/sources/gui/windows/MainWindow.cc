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

#include "MainWindow.hh"

namespace holovibes
{
	namespace gui
	{
		MainWindow::MainWindow(Holovibes& holovibes, QWidget *parent)
			: QMainWindow(parent),
			holovibes_(holovibes),
			mainDisplay(nullptr),
			sliceXZ(nullptr),
			sliceYZ(nullptr),
			displayAngle(0.f),
			xzAngle(0.f),
			yzAngle(270.f),
			displayFlip(0),
			xzFlip(0),
			yzFlip(1),
			is_enabled_camera_(false),
			is_enabled_average_(false),
			is_batch_img_(true),
			is_batch_interrupted_(false),
			z_step_(0.01f),
			camera_type_(Holovibes::NONE),
			last_contrast_type_("Magnitude"),
			plot_window_(nullptr),
			record_thread_(nullptr),
			CSV_record_thread_(nullptr),
			file_index_(1),
			gpib_interface_(nullptr),
			theme_index_(0),
			display_width_(512),
			display_height_(512),
			is_enabled_autofocus_(false)
		{
			ui.setupUi(this);
			setWindowIcon(QIcon("icon1.ico"));
			InfoManager::get_manager(findChild<GroupBox *>("InfoGroupBox"));

			move(QPoint(520, 545));

			// Hide non default tab
			findChild<GroupBox *>("PostProcessingGroupBox")->setHidden(true);
			findChild<GroupBox *>("RecordGroupBox")->setHidden(true);
			findChild<GroupBox *>("InfoGroupBox")->setHidden(true);

			findChild<QAction *>("actionSpecial")->setChecked(false);
			findChild<QAction *>("actionRecord")->setChecked(false);
			findChild<QAction *>("actionInfo")->setChecked(false);

			layout_toggled(false);

			load_ini(GLOBAL_INI_PATH);

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

			QComboBox *depth_cbox = findChild<QComboBox *>("ImportDepthComboBox");
			connect(depth_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(hide_endianess()));

			QComboBox *window_cbox = findChild<QComboBox *>("WindowSelectionComboBox");
			connect(window_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(change_window()));

			resize(width(), 425);
			// Display default values
			holovibes_.get_compute_desc().compute_mode.exchange(Computation::Stop);
			notify();
			holovibes_.get_compute_desc().compute_mode.exchange(Computation::Direct);
			notify();
		}

		MainWindow::~MainWindow()
		{
			delete z_up_shortcut_;
			delete z_down_shortcut_;
			delete p_left_shortcut_;
			delete p_right_shortcut_;
			delete autofocus_ctrl_c_shortcut_;

			close_critical_compute();
			camera_none();
			close_windows();
			remove_infos();

			holovibes_.dispose_compute();
			if (!is_direct_mode())
				holovibes_.dispose_capture();
			InfoManager::stop_display();
		}

		void MainWindow::notify()
		{
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			const bool is_direct = is_direct_mode();
			if (cd.compute_mode.load() == Computation::Stop)
			{
				findChild<GroupBox *>("ImageRenderingGroupBox")->setEnabled(false);
				findChild<GroupBox *>("ViewGroupBox")->setEnabled(false);
				findChild<GroupBox *>("PostProcessingGroupBox")->setEnabled(false);
				findChild<GroupBox *>("RecordGroupBox")->setEnabled(false);
				findChild<GroupBox *>("ImportGroupBox")->setEnabled(true);
				findChild<GroupBox *>("InfoGroupBox")->setEnabled(true);
				if (findChild<QRadioButton *>("DirectRadioButton")->isChecked())
					holovibes_.get_compute_desc().compute_mode.exchange(Computation::Direct);
				else
					holovibes_.get_compute_desc().compute_mode.exchange(Computation::Hologram);
				return;
			}
			else if (cd.compute_mode.load() == Computation::Direct && is_enabled_camera_)
			{
				findChild<GroupBox *>("ImageRenderingGroupBox")->setEnabled(true);
				findChild<GroupBox *>("RecordGroupBox")->setEnabled(true);
			}
			else if (cd.compute_mode.load() == Computation::Hologram && is_enabled_camera_)
			{
				findChild<GroupBox *>("ImageRenderingGroupBox")->setEnabled(true);
				findChild<GroupBox *>("ViewGroupBox")->setEnabled(true);
				findChild<GroupBox *>("PostProcessingGroupBox")->setEnabled(true);
				findChild<GroupBox *>("RecordGroupBox")->setEnabled(true);
			}
			findChild<QCheckBox *>("RecordFloatOutputCheckBox")->setEnabled(!is_direct);
			findChild<QCheckBox *>("RecordComplexOutputCheckBox")->setEnabled(!is_direct);

			findChild<QLineEdit *>("ROIOutputPathLineEdit")->setEnabled(!is_direct && cd.average_enabled.load());
			findChild<QToolButton *>("ROIOutputToolButton")->setEnabled(!is_direct && cd.average_enabled.load());
			findChild<QPushButton *>("ROIOutputRecPushButton")->setEnabled(!is_direct && cd.average_enabled.load());
			findChild<QPushButton *>("ROIOutputBatchPushButton")->setEnabled(!is_direct && cd.average_enabled.load());
			findChild<QPushButton *>("ROIOutputStopPushButton")->setEnabled(!is_direct && cd.average_enabled.load());
			findChild<QToolButton *>("ROIFileBrowseToolButton")->setEnabled(cd.average_enabled.load());
			findChild<QLineEdit *>("ROIFilePathLineEdit")->setEnabled(cd.average_enabled.load());
			findChild<QPushButton *>("SaveROIPushButton")->setEnabled(cd.average_enabled.load());
			findChild<QPushButton *>("LoadROIPushButton")->setEnabled(cd.average_enabled.load());

			QPushButton* signalBtn = findChild<QPushButton *>("AverageSignalPushButton");
			signalBtn->setEnabled(cd.average_enabled.load());
			signalBtn->setStyleSheet((signalBtn->isEnabled() &&
				mainDisplay->getKindOfSelection() == KindOfSelection::Signal) ? "QPushButton {color: #8E66D9;}" : "");
			// FF0080 fushia
			// 8E66D9 mauve

			QPushButton* noiseBtn = findChild<QPushButton *>("AverageNoisePushButton");
			noiseBtn->setEnabled(cd.average_enabled.load());
			noiseBtn->setStyleSheet((noiseBtn->isEnabled() &&
				mainDisplay->getKindOfSelection() == KindOfSelection::Noise) ? "QPushButton {color: #00A4AB;}" : "");
			// 428EA3 gris-turquoise
			// 00A4AB turquoise

			/*findChild<QLabel *>("AverageSignalLabel")->setText(
				cd.average_enabled.load() ? "<font color='DeepPink'>Signal</font>" : "Signal");
			findChild<QLabel *>("AverageNoiseLabel")->setText(
				cd.average_enabled.load() ? "<font color='Turquoise'>Noise</font>" : "Noise");*/

			findChild<QCheckBox*>("PhaseUnwrap2DCheckBox")->
				setEnabled(((!is_direct && (cd.view_mode.load() == ComplexViewMode::PhaseIncrease) ||
				(cd.view_mode.load() == ComplexViewMode::Argument)) ? (true) : (false)));

			findChild<QCheckBox *>("STFTCutsCheckBox")->setEnabled(!is_direct && cd.stft_enabled.load()
				&& !cd.filter_2d_enabled.load() && !cd.signal_trig_enabled.load());
			findChild<QCheckBox *>("STFTCutsCheckBox")->setChecked(!is_direct && cd.stft_view_enabled.load());

			QPushButton *filter_button = findChild<QPushButton *>("Filter2DPushButton");
			filter_button->setEnabled(!is_direct && !cd.stft_view_enabled.load() && !cd.filter_2d_enabled.load() && !cd.stft_view_enabled.load());
			filter_button->setStyleSheet((!is_direct && cd.filter_2d_enabled.load()) ? "QPushButton {color: #009FFF;}" : "");
			findChild<QPushButton *>("CancelFilter2DPushButton")->setEnabled(!is_direct && cd.filter_2d_enabled.load());

			findChild<QCheckBox *>("ContrastCheckBox")->setChecked(cd.contrast_enabled.load());
			findChild<QCheckBox *>("LogScaleCheckBox")->setChecked(!is_direct && cd.log_scale_enabled.load());
			findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")->setEnabled(!is_direct && cd.contrast_enabled.load());
			findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")->setEnabled(!is_direct && cd.contrast_enabled.load());
			findChild<QPushButton *>("AutoContrastPushButton")->setEnabled(!is_direct && cd.contrast_enabled.load());
			if (cd.current_window.load() == WindowKind::MainDisplay)
			{
				findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")
					->setValue((cd.log_scale_enabled.load()) ? cd.contrast_min.load() : log10(cd.contrast_min.load()));
				findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")
					->setValue((cd.log_scale_enabled.load()) ? cd.contrast_max.load() : log10(cd.contrast_max.load()));
			}
			else if (cd.current_window.load() == WindowKind::SliceXZ)
			{
				findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")
					->setValue((cd.log_scale_enabled.load()) ? cd.contrast_min.load() : log10(cd.contrast_min_slice_xz.load()));
				findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")
					->setValue((cd.log_scale_enabled.load()) ? cd.contrast_min.load() : log10(cd.contrast_max_slice_xz.load()));
			}
			else if (cd.current_window.load() == WindowKind::SliceYZ)
			{
				findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")
					->setValue((cd.log_scale_enabled.load()) ? cd.contrast_min.load() : log10(cd.contrast_min_slice_yz.load()));
				findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")
					->setValue((cd.log_scale_enabled.load()) ? cd.contrast_min.load() : log10(cd.contrast_max_slice_yz.load()));
			}
			findChild<QCheckBox *>("FFTShiftCheckBox")->setChecked(cd.shift_corners_enabled.load());

			QSpinBox *p_vibro = findChild<QSpinBox *>("ImageRatioPSpinBox");
			p_vibro->setEnabled(!is_direct && cd.vibrometry_enabled.load());
			p_vibro->setValue(cd.pindex.load() + 1);
			p_vibro->setMaximum(cd.nsamples.load());
			QSpinBox *q_vibro = findChild<QSpinBox *>("ImageRatioQSpinBox");
			q_vibro->setEnabled(!is_direct && cd.vibrometry_enabled.load());
			q_vibro->setValue(cd.vibrometry_q.load() + 1);
			q_vibro->setMaximum(cd.nsamples.load());

			findChild<QCheckBox*>("ImageRatioCheckBox")->setChecked(!is_direct && cd.vibrometry_enabled.load());
			findChild<QCheckBox *>("ConvoCheckBox")->setEnabled(!is_direct && cd.convo_matrix.size() == 0 ? false : true);
			findChild<QCheckBox *>("AverageCheckBox")->setChecked(!is_direct && cd.average_enabled.load());
			findChild<QCheckBox *>("FlowgraphyCheckBox")->setChecked(!is_direct && cd.flowgraphy_enabled.load());
			findChild<QSpinBox *>("FlowgraphyLevelSpinBox")->setEnabled(!is_direct && cd.flowgraphy_level.load());
			findChild<QSpinBox *>("FlowgraphyLevelSpinBox")->setValue(cd.flowgraphy_level.load());

			findChild<QPushButton *>("AutofocusRunPushButton")->setEnabled(!is_direct && cd.algorithm.load() != Algorithm::None);
			findChild<QLabel *>("AutofocusLabel")->setText((is_enabled_autofocus_) ? "<font color='Yellow'>Autofocus:</font>" : "Autofocus:");
			findChild<QCheckBox *>("STFTCheckBox")->setEnabled(!is_direct && !cd.stft_view_enabled.load() && !cd.signal_trig_enabled.load());
			findChild<QCheckBox *>("STFTCheckBox")->setChecked(!is_direct && cd.stft_enabled.load());
			findChild<QSpinBox *>("STFTStepsSpinBox")->setEnabled(!is_direct);
			findChild<QSpinBox *>("STFTStepsSpinBox")->setValue(cd.stft_steps.load());
			findChild<QPushButton *>("TakeRefPushButton")->setEnabled(!is_direct && !cd.ref_sliding_enabled.load());
			findChild<QPushButton *>("SlidingRefPushButton")->setEnabled(!is_direct && !cd.ref_diff_enabled.load() && !cd.ref_sliding_enabled.load());
			findChild<QPushButton *>("CancelRefPushButton")->setEnabled(!is_direct && (cd.ref_diff_enabled.load() || cd.ref_sliding_enabled.load()));
			findChild<QComboBox *>("AlgorithmComboBox")->setEnabled(!is_direct);
			findChild<QComboBox *>("AlgorithmComboBox")->setCurrentIndex(cd.algorithm.load());
			findChild<QComboBox *>("ViewModeComboBox")->setCurrentIndex(cd.view_mode.load());
			findChild<QPushButton *>("SetPhaseNumberPushButton")->setEnabled(!is_direct && !cd.stft_view_enabled.load());
			findChild<QLineEdit *>("PhaseNumberLineEdit")->setEnabled(!is_direct);
			findChild<QLineEdit *>("PhaseNumberLineEdit")->setText(QString::fromUtf8(std::to_string(cd.nsamples.load()).c_str()));
			findChild<QSpinBox *>("PSpinBox")->setEnabled(!is_direct);
			findChild<QSpinBox *>("PSpinBox")->setValue(cd.pindex.load() + 1);
			findChild<QSpinBox *>("PSpinBox")->setMaximum(cd.nsamples.load());
			findChild<QDoubleSpinBox *>("WaveLengthDoubleSpinBox")->setEnabled(!is_direct);
			findChild<QDoubleSpinBox *>("WaveLengthDoubleSpinBox")->setValue(cd.lambda.load() * 1.0e9f);
			findChild<QDoubleSpinBox *>("ZDoubleSpinBox")->setEnabled(!is_direct);
			findChild<QDoubleSpinBox *>("ZDoubleSpinBox")->setValue(cd.zdistance.load());
			findChild<QDoubleSpinBox*>("ZDoubleSpinBox")->setSingleStep(z_step_);
			findChild<QDoubleSpinBox *>("ZStepDoubleSpinBox")->setEnabled(!is_direct);
			findChild<QDoubleSpinBox *>("PixelSizeDoubleSpinBox")->setEnabled(!is_direct && !cd.is_cine_file.load());
			findChild<QDoubleSpinBox *>("PixelSizeDoubleSpinBox")->setValue(cd.import_pixel_size.load());
			findChild<QLineEdit *>("BoundaryLineEdit")->setEnabled(!is_direct);
			findChild<QLineEdit *>("BoundaryLineEdit")->clear();
			findChild<QCheckBox *>("ImgAccuCheckBox")->setChecked(!is_direct && cd.img_acc_enabled.load());
			findChild<QSpinBox *>("ImgAccuSpinBox")->setValue(cd.img_acc_level.load());
			findChild<QSpinBox *>("KernelBufferSizeSpinBox")->setValue(cd.special_buffer_size.load());
			findChild<QDoubleSpinBox *>("AutofocusZMaxDoubleSpinBox")->setValue(cd.autofocus_z_max.load());
			findChild<QDoubleSpinBox *>("AutofocusZMinDoubleSpinBox")->setValue(cd.autofocus_z_min.load());
			findChild<QSpinBox *>("AutofocusStepsSpinBox")->setValue(cd.autofocus_z_div.load());
			findChild<QSpinBox *>("AutofocusLoopsSpinBox")->setValue(cd.autofocus_z_iter.load());
			findChild<QCheckBox *>("CineFileCheckBox")->setChecked(cd.is_cine_file.load());
			findChild<QSpinBox *>("ImportWidthSpinBox")->setEnabled(!cd.is_cine_file.load());
			findChild<QSpinBox *>("ImportHeightSpinBox")->setEnabled(!cd.is_cine_file.load());
			findChild<QComboBox *>("ImportDepthComboBox")->setEnabled(!cd.is_cine_file.load());
			QString depth_value = findChild<QComboBox *>("ImportDepthComboBox")->currentText();
			findChild<QComboBox *>("ImportEndiannessComboBox")->setEnabled(depth_value == "16" && !cd.is_cine_file.load());
			findChild<QCheckBox *>("ExtTrigCheckBox")->setEnabled(cd.signal_trig_enabled.load());
		}

		void MainWindow::notify_error(std::exception& e, const char* msg)
		{
			CustomException* err_ptr = dynamic_cast<CustomException*>(&e);
			std::string str;
			if (err_ptr != nullptr)
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				if (err_ptr->get_kind() == error_kind::fail_update)
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
					}
				}
				if (err_ptr->get_kind() == error_kind::fail_accumulation)
				{
					cd.img_acc_enabled.exchange(false);
					cd.img_acc_level.exchange(1);
				}
				close_critical_compute();

				str = "GPU allocation error occured.\nCuda error message\n" + std::string(msg);
				display_error(str);
			}
			else
			{
				str = "Unknown error occured.";
				display_error(str);
			}
			notify();
		}

		void MainWindow::display_message(QString msg)
		{
			InfoManager::get_manager()->remove_info("Message");
			InfoManager::get_manager()->update_info("Message", msg.toStdString());
		}

		void MainWindow::layout_toggled(bool b)
		{
			uint childCount = 0;
			std::vector<GroupBox *> v;

			v.push_back(findChild<GroupBox *>("ImageRenderingGroupBox"));
			v.push_back(findChild<GroupBox *>("ViewGroupBox"));
			v.push_back(findChild<GroupBox *>("PostProcessingGroupBox"));
			v.push_back(findChild<GroupBox *>("RecordGroupBox"));
			v.push_back(findChild<GroupBox *>("ImportGroupBox"));
			v.push_back(findChild<GroupBox *>("InfoGroupBox"));

			for each (GroupBox *var in v)
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

		void MainWindow::camera_ids()
		{
			change_camera(Holovibes::IDS);
		}

		void MainWindow::camera_ixon()
		{
			change_camera(Holovibes::IXON);
		}

		void MainWindow::camera_none()
		{
			if (!is_direct_mode())
				holovibes_.dispose_compute();
			holovibes_.dispose_capture();
			close_windows();
			remove_infos();
			findChild<QAction*>("actionSettings")->setEnabled(false);
			holovibes_.get_compute_desc().compute_mode.exchange(Computation::Stop);
			notify();
		}

		void MainWindow::camera_adimec()
		{
			change_camera(Holovibes::ADIMEC);
		}

		void MainWindow::camera_edge()
		{
			change_camera(Holovibes::EDGE);
		}

		void MainWindow::camera_pike()
		{
			change_camera(Holovibes::PIKE);
		}

		void MainWindow::camera_pixelfly()
		{
			change_camera(Holovibes::PIXELFLY);
		}

		void MainWindow::camera_xiq()
		{
			change_camera(Holovibes::XIQ);
		}

		void MainWindow::credits()
		{
			std::string msg =
				"Holovibes " + version + "\n\n"

				"Developers:\n\n"

				"Thomas Jarrossay\n"
				"Alexandre Bartz\n"

				"Cyril Cetre\n"
				"Clement Ledant\n"

				"Eric Delanghe\n"
				"Arnaud Gaillard\n"
				"Geoffrey Le Gourrierec\n"

				"Jeffrey Bencteux\n"
				"Thomas Kostas\n"
				"Pierre Pagnoux\n"

				"Antoine Dillée\n"
				"Romain Cancillière\n"

				"Michael Atlan\n";
			QMessageBox msg_box;
			msg_box.setText(QString::fromUtf8(msg.c_str()));
			msg_box.setIcon(QMessageBox::Information);
			msg_box.exec();
		}

		void MainWindow::configure_camera()
		{
			open_file(boost::filesystem::current_path().generic_string() + "/" + holovibes_.get_camera_ini_path());
		}

		void MainWindow::init_image_mode(QPoint& position, QSize& size)
		{
			holovibes_.dispose_compute();

			if (mainDisplay)
			{
				position = mainDisplay->framePosition();
				size = mainDisplay->size();
				mainDisplay.reset(nullptr);
			}
		}

		void MainWindow::remove_infos()
		{
			try
			{
				auto manager = InfoManager::get_manager();
				manager->remove_info("Input FPS");
				manager->remove_info("Info");
				manager->remove_info("Message");
				manager->remove_info("Error");
				manager->remove_info("InputQueue");
				manager->remove_info("OutputQueue");
				manager->remove_info("Rendering FPS");
				manager->remove_info("STFTQueue");
				manager->remove_info("STFT Slice Cursor");
				manager->remove_info("Status");
				manager->update_info("ImgSource", "None");
			}
			catch (std::exception& e)
			{
				std::cerr << e.what() << std::endl;
			}
		}

		void MainWindow::close_windows()
		{
			if (sliceXZ)
				sliceXZ.reset(nullptr);
			if (sliceYZ)
				sliceYZ.reset(nullptr);
			if (plot_window_)
				plot_window_.reset(nullptr);
			if (mainDisplay)
				mainDisplay.reset(nullptr);
		}

		void MainWindow::set_direct_mode()
		{
			close_critical_compute();
			close_windows();
			holovibes_.get_compute_desc().compute_mode.exchange(Computation::Stop);
			notify();
			if (is_enabled_camera_)
			{
				QPoint pos(0, 4);
				QSize size(512, 512);
				init_image_mode(pos, size);
				holovibes_.get_compute_desc().compute_mode.exchange(Computation::Direct);
				mainDisplay.reset(new DirectWindow(
					pos,
					size,
					holovibes_.get_capture_queue()));
				set_convolution_mode(false);
				notify();
			}
		}

		void MainWindow::set_holographic_mode()
		{
			close_critical_compute();
			close_windows();
			if (is_enabled_camera_)
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.compute_mode.exchange(Computation::Hologram);
				QPoint pos(0, 4);
				QSize size(512, 512);
				init_image_mode(pos, size);
				uint depth = 2;
				if (cd.view_mode.load() == ComplexViewMode::Complex)
				{
					last_contrast_type_ = "Complex output";
					depth = 8;
				}
				try
				{
					cd.nsamples.exchange(1);
					holovibes_.init_compute(ThreadCompute::PipeType::PIPE, depth);
					while (!holovibes_.get_pipe());
					holovibes_.get_pipe()->register_observer(*this);
					setPhase();
					holovibes_.get_pipe()->request_update_n(1);
					mainDisplay.reset(new HoloWindow(
						pos,
						size,
						holovibes_.get_output_queue(),
						holovibes_.get_pipe(),
						holovibes_.get_compute_desc()));
					mainDisplay->setAngle(displayAngle);
					mainDisplay->setFlip(displayFlip);
					//if (!cd.flowgraphy_enabled && !is_direct_mode())
						//holovibes_.get_pipe()->request_autocontrast();
					cd.contrast_enabled.exchange(true);
					if (!cd.flowgraphy_enabled.load())
						set_auto_contrast();
					notify();
				}
				catch (std::exception& e)
				{
					mainDisplay.reset(nullptr);
					display_error(e.what());
				}
			}
		}

		void MainWindow::set_complex_mode(bool value)
		{
			close_critical_compute();
			close_windows();
			QPoint pos(0, 0);
			QSize size(512, 512);
			init_image_mode(pos, size);
			uint depth = 2;
			try
			{
				if (value == true)
					depth = 8;
				holovibes_.init_compute(ThreadCompute::PipeType::PIPE, depth);
				while (!holovibes_.get_pipe());
				holovibes_.get_pipe()->register_observer(*this);
				mainDisplay.reset(new HoloWindow(
					pos,
					size,
					holovibes_.get_output_queue(),
					holovibes_.get_pipe(),
					holovibes_.get_compute_desc()));
				mainDisplay->setAngle(displayAngle);
				mainDisplay->setFlip(displayFlip);
				notify();
			}
			catch (std::exception& e)
			{
				mainDisplay.reset(nullptr);
				display_error(e.what());
			}
		}

		void MainWindow::set_convolution_mode(const bool value)
		{
			if (value == true && holovibes_.get_compute_desc().convo_matrix.empty())
			{
				display_error("No valid kernel has been given");
				holovibes_.get_compute_desc().convolution_enabled.exchange(false);
			}
			else
			{
				holovibes_.get_compute_desc().convolution_enabled.exchange(value);
				if (!holovibes_.get_compute_desc().flowgraphy_enabled && !is_direct_mode())
					set_auto_contrast();
			}
			notify();
		}

		void MainWindow::set_flowgraphy_mode(const bool value)
		{
			holovibes_.get_compute_desc().flowgraphy_enabled.exchange(value);
			if (!is_direct_mode())
				pipe_refresh();
			notify();
		}

		bool MainWindow::is_direct_mode()
		{
			return (holovibes_.get_compute_desc().compute_mode.load() == Computation::Direct);
		}

		void MainWindow::set_image_mode()
		{
			if (holovibes_.get_compute_desc().compute_mode.load() == Computation::Direct)
				set_direct_mode();
			if (holovibes_.get_compute_desc().compute_mode.load() == Computation::Hologram)
				set_holographic_mode();
		}

		void MainWindow::reset()
		{
			Config&	config = global::global_config;
			int					device = 0;

			close_critical_compute();
			camera_none();
			auto manager = InfoManager::get_manager();
			manager->update_info("Status", "Resetting...");
			qApp->processEvents();
			mainDisplay.reset(nullptr);
			if (!is_direct_mode())
				holovibes_.dispose_compute();
			holovibes_.dispose_capture();
			is_enabled_camera_ = false;
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
			close_windows();
			change_camera(camera_type_);
			load_ini(GLOBAL_INI_PATH);
			remove_infos();
			notify();
		}

		void MainWindow::take_reference()
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.ref_diff_enabled.exchange(true);
				holovibes_.get_pipe()->request_ref_diff_refresh();
				InfoManager::update_info("Reference", "Processing... ");
				notify();
			}
		}

		void MainWindow::take_sliding_ref()
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.ref_sliding_enabled.exchange(true);
				holovibes_.get_pipe()->request_ref_diff_refresh();
				InfoManager::update_info("Reference", "Processing...");
				notify();
			}
		}

		void MainWindow::cancel_take_reference()
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.ref_diff_enabled.exchange(false);
				cd.ref_sliding_enabled.exchange(false);
				holovibes_.get_pipe()->request_ref_diff_refresh();
				InfoManager::remove_info("Reference");
				notify();
			}
		}

		void MainWindow::set_filter2D()
		{
			if (!is_direct_mode())
			{
				mainDisplay->resetTransform();
				mainDisplay->setKindOfSelection(KindOfSelection::Filter2D);

				findChild<QPushButton*>("Filter2DPushButton")->setStyleSheet("QPushButton {color: #009FFF;}");
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.log_scale_enabled.exchange(true);
				cd.shift_corners_enabled.exchange(false);
				cd.filter_2d_enabled.exchange(true);
				InfoManager::get_manager()->update_info("Filter2D", "Processing...");
				set_auto_contrast();
				notify();
			}
		}

		void MainWindow::cancel_filter2D()
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				InfoManager::get_manager()->remove_info("Filter2D");
				cd.filter_2d_enabled.exchange(false);
				cd.log_scale_enabled.exchange(false);
				cd.stftRoiZone(Rectangle(0, 0), AccessMode::Set);
				mainDisplay->setKindOfSelection(KindOfSelection::Zoom);
				set_auto_contrast();
				notify();
			}
		}

		void MainWindow::setPhase()
		{
			if (!is_direct_mode())
			{
				QLineEdit* lineEdit = findChild<QLineEdit *>("PhaseNumberLineEdit");
				int phaseNumber = lineEdit->text().toInt();
				phaseNumber = (phaseNumber <= 0) ? 1 : phaseNumber;
				ComputeDescriptor&	cd = holovibes_.get_compute_desc();
				Queue&				in = holovibes_.get_capture_queue();

				if (cd.stft_enabled.load()
					|| phaseNumber <= static_cast<int>(in.get_max_elts()))
				{
					holovibes_.get_pipe()->request_update_n(phaseNumber);
					//if (cd.stft_view_enabled.load())
					//{
						//_win_stft_0->resize((value > 120 ? value : 120) * 2, gl_window_->height());
						//gl_win_stft_1->resize(gl_window_->width(), (value > 120 ? value : 120) * 2);
						//holovibes_.get_pipe()->update_stft_slice_queue();

						//stft_view(false);
						//std::this_thread::sleep_for(std::chrono::milliseconds(10));
						//stft_view(true);
					//}
					notify();
				}
				else
				{
					lineEdit->setText(
						QString::fromUtf8(
							std::to_string(static_cast<const int>(in.get_max_elts())).c_str()));
				}
			}
		}

		void MainWindow::set_special_buffer_size(int value)
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.special_buffer_size.exchange(value);
				if (cd.special_buffer_size.load() < static_cast<std::atomic<int>>(cd.flowgraphy_level.load()))
				{
					if (cd.special_buffer_size.load() % 2 == 0)
						cd.flowgraphy_level.exchange(cd.special_buffer_size.load() - 1);
					else
						cd.flowgraphy_level.exchange(cd.special_buffer_size.load());
				}
				notify();
				if (!holovibes_.get_compute_desc().flowgraphy_enabled)
					set_auto_contrast();
				notify();
			}
		}

		void MainWindow::set_p(int value)
		{
			value--;
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();

				if (value < static_cast<int>(cd.nsamples.load()))
				{
					cd.pindex.exchange(value);
					if (!holovibes_.get_compute_desc().flowgraphy_enabled && !is_direct_mode())
						set_auto_contrast();
					notify();
				}
				else
					display_error("p param has to be between 0 and n");
			}
		}

		void MainWindow::set_flowgraphy_level(const int value)
		{
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			int flag = 0;

			if (!is_direct_mode())
			{
				if (value % 2 == 0)
				{
					if (value + 1 <= cd.special_buffer_size.load())
					{
						cd.flowgraphy_level.exchange(value + 1);
						flag = 1;
					}
				}
				else
				{
					if (value <= cd.special_buffer_size.load())
					{
						cd.flowgraphy_level.exchange(value);
						flag = 1;
					}
				}
				notify();
				if (flag == 1)
					pipe_refresh();
			}
		}

		void MainWindow::increment_p()
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();

				if (cd.pindex.load() < cd.nsamples.load())
				{
					cd.pindex.exchange(cd.pindex.load() + 1);
					notify();
					if (!holovibes_.get_compute_desc().flowgraphy_enabled)
						set_auto_contrast();
				}
				else
					display_error("p param has to be between 1 and n");
				notify();
			}
		}

		void MainWindow::decrement_p()
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();

				if (cd.pindex.load() > 0)
				{
					cd.pindex.exchange(cd.pindex.load() - 1);
					if (!holovibes_.get_compute_desc().flowgraphy_enabled)
						set_auto_contrast();
					notify();
				}
				else
					display_error("p param has to be between 1 and n");
			}
		}

		void MainWindow::set_wavelength(const double value)
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.lambda.exchange(static_cast<float>(value) * 1.0e-9f);
				pipe_refresh();

				// Updating the GUI
				findChild<QLineEdit*>("BoundaryLineEdit")->clear();
				findChild<QLineEdit*>("BoundaryLineEdit")->insert(QString::number(holovibes_.get_boundary()));
				notify();
			}
		}

		void MainWindow::set_z(const double value)
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.zdistance.exchange(static_cast<float>(value));
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::increment_z()
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				set_z(cd.zdistance.load() + z_step_);
				notify();
			}
		}

		void MainWindow::decrement_z()
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				set_z(cd.zdistance.load() - z_step_);
				notify();
			}
		}

		void MainWindow::set_z_step(const double value)
		{
			z_step_ = value;
			notify();
		}

		void MainWindow::set_algorithm(const QString value)
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();

				//mainDisplay->setKindOfSelection(KindOfSelection::Zoom);	// raw Zoom tmp

				if (value == "None")
					cd.algorithm.exchange(Algorithm::None);
				else if (value == "1FFT")
					cd.algorithm.exchange(Algorithm::FFT1);
				else if (value == "2FFT")
					cd.algorithm.exchange(Algorithm::FFT2);
				else
					assert(!"Unknow Algorithm.");
				if (!holovibes_.get_compute_desc().flowgraphy_enabled)
					set_auto_contrast();
				notify();
			}
		}

		void MainWindow::set_stft(bool b)
		{
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			if (!is_direct_mode())
			{
				cd.nsamples.exchange(cd.stft_level.load());
				cd.stft_level.exchange(cd.nsamples.load());
				cd.stft_enabled.exchange(b);
				holovibes_.get_pipe()->request_update_n(cd.nsamples.load());
				notify();
			}
		}

		void MainWindow::update_stft_steps(int value)
		{
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			if (!is_direct_mode())
			{
				cd.stft_steps.exchange(value);
				notify();
			}
		}

		void MainWindow::cancel_stft_slice_view()
		{
			ComputeDescriptor&	cd = holovibes_.get_compute_desc();
			auto manager = InfoManager::get_manager();

			//set_contrast_mode(false);
			cd.stft_view_enabled.exchange(false);
			manager->remove_info("STFT Slice Cursor");

			holovibes_.get_pipe()->delete_stft_slice_queue();
			while (holovibes_.get_pipe()->get_cuts_delete_request());
			sliceXZ.reset(nullptr);
			sliceYZ.reset(nullptr);

			findChild<QCheckBox*>("STFTCutsCheckBox")->setChecked(false);
			findChild<QCheckBox*>("STFTCheckBox")->setEnabled(true);

			mainDisplay->setCursor(Qt::ArrowCursor);
			mainDisplay->setKindOfSelection(KindOfSelection::Zoom);

			notify();
		}

		void MainWindow::stft_view(bool checked)
		{
			ComputeDescriptor&	cd = holovibes_.get_compute_desc();
			auto manager = InfoManager::get_manager();
			manager->update_info("STFT Slice Cursor", "(Y,X) = (0,0)");
			if (checked)
			{
				try
				{
					if (cd.filter_2d_enabled.load())
						cancel_filter2D();
					holovibes_.get_pipe()->create_stft_slice_queue();
					// set positions of new windows according to the position of the main GL window
					QPoint			xzPos = mainDisplay->framePosition() + QPoint(0, mainDisplay->height() + 27);
					QPoint			yzPos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 8, 0);
					const ushort	nImg = cd.nsamples.load();
					const uint		nSize = (nImg < 128 ? 128 : nImg) * 2;

					while (holovibes_.get_pipe()->get_cuts_request());
					sliceXZ.reset(nullptr);
					sliceXZ.reset(new SliceWindow(
						xzPos,
						QSize(mainDisplay->width(), nSize),
						holovibes_.get_pipe()->get_stft_slice_queue(0)));
					sliceXZ->setTitle("Slice XZ");
					sliceXZ->setAngle(xzAngle);
					sliceXZ->setFlip(xzFlip);

					sliceYZ.reset(nullptr);
					sliceYZ.reset(new SliceWindow(
						yzPos,
						QSize(nSize, mainDisplay->height()),
						holovibes_.get_pipe()->get_stft_slice_queue(1)));
					sliceYZ->setTitle("Slice YZ");
					sliceYZ->setAngle(yzAngle);
					sliceYZ->setFlip(yzFlip);

					mainDisplay->setKindOfSelection(KindOfSelection::SliceZoom);

					findChild<QCheckBox*>("STFTCutsCheckBox")->setChecked(true);
					cd.stft_view_enabled.exchange(true);
					notify();
				}
				catch (std::logic_error& e)
				{
					std::cerr << e.what() << std::endl;
					cancel_stft_slice_view();
				}
			}
			else
			{
				cancel_stft_slice_view();
			}
		}

		void MainWindow::set_view_mode(const QString value)
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();

				bool pipeline_checked = false;

				std::cout << "Value = " << value.toUtf8().constData() << std::endl;
				if (last_contrast_type_ == "Complex output" && value != "Complex output")
				{
					set_complex_mode(false);
				}
				if (value == "Magnitude")
				{
					cd.view_mode.exchange(ComplexViewMode::Modulus);
					last_contrast_type_ = value;
				}
				else if (value == "Squared magnitude")
				{
					cd.view_mode.exchange(ComplexViewMode::SquaredModulus);
					last_contrast_type_ = value;
				}
				else if (value == "Argument")
				{
					cd.view_mode.exchange(ComplexViewMode::Argument);
					last_contrast_type_ = value;
				}
				else if (value == "Complex output")
				{
					set_complex_mode(true);
					cd.view_mode.exchange(ComplexViewMode::Complex);
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
							cd.view_mode.exchange(ComplexViewMode::PhaseIncrease);
					}
				}
				if (!holovibes_.get_compute_desc().flowgraphy_enabled)
					set_auto_contrast();

				//set_enable_unwrap_box();
				notify();
			}
		}

		void MainWindow::rotateTexture()
		{
			QComboBox *c = findChild<QComboBox*>("WindowSelectionComboBox");
			QString s = c->currentText();

			if (s == QString("mainDisplay"))
			{
				displayAngle = (displayAngle == 270.f) ? 0.f : displayAngle + 90.f;
				mainDisplay->setAngle(displayAngle);
			}
			else if (s == QString("sliceXZ") && sliceXZ)
			{
				xzAngle = (xzAngle == 270.f) ? 0.f : xzAngle + 90.f;
				sliceXZ->setAngle(xzAngle);
			}
			else if (s == QString("sliceYZ") && sliceYZ)
			{
				yzAngle = (yzAngle == 270.f) ? 0.f : yzAngle + 90.f;
				sliceYZ->setAngle(yzAngle);
			}
			notify();
		}

		void MainWindow::flipTexture()
		{
			QComboBox *c = findChild<QComboBox*>("WindowSelectionComboBox");
			QString s = c->currentText();

			if (s == QString("mainDisplay"))
			{
				displayFlip = !displayFlip;
				mainDisplay->setFlip(displayFlip);
			}
			else if (s == QString("sliceXZ") && sliceXZ)
			{
				xzFlip = !xzFlip;
				sliceXZ->setFlip(xzFlip);
			}
			else if (s == QString("sliceYZ") && sliceYZ)
			{
				yzFlip = !yzFlip;
				sliceYZ->setFlip(yzFlip);
			}
			notify();
		}

		void MainWindow::set_unwrap_history_size(int value)
		{
			if (!is_direct_mode())
			{
				holovibes_.get_compute_desc().unwrap_history_size = value;
				holovibes_.get_pipe()->request_update_unwrap_size(value);
				notify();
			}
		}

		void MainWindow::set_unwrapping_1d(const bool value)
		{
			if (!is_direct_mode())
			{
				holovibes_.get_pipe()->request_unwrapping_1d(value);
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_unwrapping_2d(const bool value)
		{
			if (!is_direct_mode())
			{
				holovibes_.get_pipe()->request_unwrapping_2d(value);
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_accumulation(bool value)
		{
			if (!is_direct_mode())
			{
				holovibes_.get_compute_desc().img_acc_enabled.exchange(value);
				// if (!value)
				holovibes_.get_pipe()->request_acc_refresh();
				// else
				//  pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_accumulation_level(int value)
		{
			if (!is_direct_mode())
			{
				holovibes_.get_compute_desc().img_acc_level.exchange(value);
				holovibes_.get_pipe()->request_acc_refresh();
				notify();
			}
		}

		void MainWindow::set_z_min(const double value)
		{
			if (!is_direct_mode())
				holovibes_.get_compute_desc().autofocus_z_min.exchange(value);
			notify();
		}

		void MainWindow::set_z_max(const double value)
		{
			if (!is_direct_mode())
				holovibes_.get_compute_desc().autofocus_z_max.exchange(value);
			notify();
		}

		void MainWindow::set_import_pixel_size(const double value)
		{
			holovibes_.get_compute_desc().import_pixel_size.exchange(value);
		}

		void MainWindow::set_z_iter(const int value)
		{
			if (!is_direct_mode())
				holovibes_.get_compute_desc().autofocus_z_iter.exchange(value);
			notify();
		}

		void MainWindow::set_z_div(const int value)
		{
			if (!is_direct_mode())
				holovibes_.get_compute_desc().autofocus_z_div.exchange(value);
			notify();
		}

		void MainWindow::set_autofocus_mode()
		{
			const float	z_max = findChild<QDoubleSpinBox*>("AutofocusZMaxDoubleSpinBox")->value();
			const float	z_min = findChild<QDoubleSpinBox*>("AutofocusZMinDoubleSpinBox")->value();
			const uint	z_div = findChild<QSpinBox*>("AutofocusStepsSpinBox")->value();
			const uint	z_iter = findChild<QSpinBox*>("AutofocusLoopsSpinBox")->value();
			ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (cd.stft_enabled.load())
				display_error("You can't call autofocus in stft mode.");
			else if (z_min < z_max)
			{
				is_enabled_autofocus_ = true;
				mainDisplay->setKindOfSelection(KindOfSelection::Autofocus);
				mainDisplay->resetTransform();
				InfoManager::get_manager()->update_info("Status", "Autofocus processing...");
				cd.autofocus_z_min.exchange(z_min);
				cd.autofocus_z_max.exchange(z_max);
				cd.autofocus_z_div.exchange(z_div);
				cd.autofocus_z_iter.exchange(z_iter);

				notify();
				is_enabled_autofocus_ = false;
			}
			else
				display_error("z min have to be strictly inferior to z max");
		}

		void MainWindow::request_autofocus_stop()
		{
			// Ctrl + C shortcut
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
			if (!is_direct_mode())
			{
				change_window();
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.contrast_enabled.exchange(value);
				set_contrast_min(findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")->value());
				set_contrast_max(findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")->value());
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::pipe_refresh()
		{
			if (!is_direct_mode())
			{
				try
				{
					holovibes_.get_pipe()->request_refresh();
				}
				catch (std::runtime_error& e)
				{
					std::cerr << e.what() << std::endl;
				}
			}
		}

		void MainWindow::set_auto_contrast()
		{
			if (!is_direct_mode())
			{
				try
				{
					holovibes_.get_pipe()->request_autocontrast();
					notify();
				}
				catch (std::runtime_error& e)
				{
					std::cerr << e.what() << std::endl;
				}
			}
		}

		void MainWindow::set_contrast_min(const double value)
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();

				if (cd.contrast_enabled.load())
				{
					if (cd.log_scale_enabled.load())
						cd.contrast_min.exchange(value);
					else
					{
						if (cd.current_window.load() == WindowKind::MainDisplay)
							cd.contrast_min.exchange(pow(10, value));
						else if (cd.current_window.load() == WindowKind::SliceXZ)
							cd.contrast_min_slice_xz.exchange(pow(10, value));
						else if (cd.current_window.load() == WindowKind::SliceYZ)
							cd.contrast_min_slice_yz.exchange(pow(10, value));
					}
					pipe_refresh();
				}
			}
		}

		void MainWindow::set_contrast_max(const double value)
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();

				if (cd.contrast_enabled.load())
				{
					if (cd.log_scale_enabled.load())
						cd.contrast_max.exchange(value);
					else
					{
						if (cd.current_window.load() == WindowKind::MainDisplay)
							cd.contrast_max.exchange(pow(10, value));
						else if (cd.current_window.load() == WindowKind::SliceXZ)
							cd.contrast_max_slice_xz.exchange(pow(10, value));
						else if (cd.current_window.load() == WindowKind::SliceYZ)
							cd.contrast_max_slice_yz.exchange(pow(10, value));
					}
					pipe_refresh();
				}
			}
		}

		void MainWindow::set_log_scale(const bool value)
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				cd.log_scale_enabled.exchange(value);

				if (cd.contrast_enabled.load())
				{
					set_contrast_min(findChild<QDoubleSpinBox*>("ContrastMinDoubleSpinBox")->value());
					set_contrast_max(findChild<QDoubleSpinBox*>("ContrastMaxDoubleSpinBox")->value());
				}
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_shifted_corners(const bool value)
		{
			if (!is_direct_mode())
			{
				holovibes_.get_compute_desc().shift_corners_enabled.exchange(value);
				pipe_refresh();
			}
		}

		void MainWindow::set_vibro_mode(const bool value)
		{
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();
				if (cd.pindex.load() > cd.nsamples.load())
					cd.pindex.exchange(cd.nsamples.load());
				if (cd.vibrometry_q.load() > cd.nsamples.load())
					cd.vibrometry_q.exchange(cd.nsamples.load());
				cd.vibrometry_enabled.exchange(value);
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_p_vibro(int value)
		{
			value--;
			if (!is_direct_mode())
			{
				ComputeDescriptor& cd = holovibes_.get_compute_desc();

				if (value < static_cast<int>(cd.nsamples.load()) && value >= 0)
				{
					cd.pindex.exchange(value);
					pipe_refresh();
					notify();
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
				ComputeDescriptor& cd = holovibes_.get_compute_desc();

				if (value < static_cast<int>(cd.nsamples.load()) && value >= 0)
				{
					holovibes_.get_compute_desc().vibrometry_q.exchange(value);
					pipe_refresh();
				}
				else
					display_error("q param has to be between 1 and phase #");
			}
		}

		void MainWindow::set_average_mode(const bool value)
		{
			holovibes_.get_compute_desc().average_enabled.exchange(value);
			mainDisplay->resetTransform();
			mainDisplay->setKindOfSelection((value) ?
				KindOfSelection::Signal : KindOfSelection::Zoom);
			if (!value)
				mainDisplay->resetSelection();
			is_enabled_average_ = value;
			notify();
		}

		void MainWindow::activeSignalZone()
		{
			mainDisplay->setKindOfSelection(KindOfSelection::Signal);
			notify();
		}

		void MainWindow::activeNoiseZone()
		{
			mainDisplay->setKindOfSelection(KindOfSelection::Noise);
			notify();
		}

		void MainWindow::set_average_graphic()
		{
			PlotWindow *plot_window = new PlotWindow(holovibes_.get_average_queue(), "ROI Average");

			connect(plot_window, SIGNAL(closed()), this, SLOT(dispose_average_graphic()), Qt::UniqueConnection);
			holovibes_.get_pipe()->request_average(&holovibes_.get_average_queue());
			pipe_refresh();
			plot_window_.reset(plot_window);
		}

		void MainWindow::dispose_average_graphic()
		{
			holovibes_.get_pipe()->request_average_stop();
			holovibes_.get_average_queue().clear();
			plot_window_.reset(nullptr);
			pipe_refresh();
		}

		void MainWindow::browse_roi_file()
		{
			QString filename = QFileDialog::getSaveFileName(this,
				tr("ROI output file"), "C://", tr("Ini files (*.ini)"));

			QLineEdit* roi_output_line_edit = findChild<QLineEdit *>("ROIFilePathLineEdit");
			roi_output_line_edit->clear();
			roi_output_line_edit->insert(filename);
		}

		void MainWindow::browse_convo_matrix_file()
		{
			QString filename = QFileDialog::getOpenFileName(this,
				tr("Matrix file"), "C://", tr("Txt files (*.txt)"));

			QLineEdit* matrix_output_line_edit = findChild<QLineEdit *>("ConvoMatrixPathLineEdit");
			matrix_output_line_edit->clear();
			matrix_output_line_edit->insert(filename);
		}

		void MainWindow::browse_roi_output_file()
		{
			QString filename = QFileDialog::getSaveFileName(this,
				tr("ROI output file"), "C://", tr("Text files (*.txt);;CSV files (*.csv)"));

			QLineEdit* roi_output_line_edit = findChild<QLineEdit *>("ROIOutputPathLineEdit");
			roi_output_line_edit->clear();
			roi_output_line_edit->insert(filename);
		}

		void MainWindow::save_roi()
		{
			QLineEdit* path_line_edit = findChild<QLineEdit *>("ROIFileLineEdit");
			std::string path = path_line_edit->text().toUtf8();

			if (path != "")
			{
				boost::property_tree::ptree ptree;

				//const GLWidget& gl_widget = gl_window_->get_gl_widget();
				/*const gui::Rectangle& signal = gl_widget.get_signal_selection();
				const gui::Rectangle& noise = gl_widget.get_noise_selection();

				ptree.put("signal.top_left_x", signal.topLeft().x());
				ptree.put("signal.top_left_y", signal.topLeft().y());
				ptree.put("signal.bottom_right_x", signal.bottomRight().x());
				ptree.put("signal.bottom_right_y", signal.bottomRight().y());

				ptree.put("noise.top_left_x", noise.topLeft().x());
				ptree.put("noise.top_left_y", noise.topLeft().y());
				ptree.put("noise.bottom_right_x", noise.bottomRight().x());
				ptree.put("noise.bottom_right_y", noise.bottomRight().y());

				boost::property_tree::write_ini(path, ptree);
				display_info("Roi saved in " + path);*/
			}
			else
				display_error("Invalid path");
		}

		void MainWindow::load_roi()
		{
			QLineEdit* path_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
			const std::string path = path_line_edit->text().toUtf8();
			boost::property_tree::ptree ptree;

			try
			{
				boost::property_tree::ini_parser::read_ini(path, ptree);

				Rectangle rectSignal;
				Rectangle rectNoise;

				rectSignal.setTopLeft(
					QPoint(ptree.get<int>("signal.top_left_x", 0),
						ptree.get<int>("signal.top_left_y", 0)));
				rectSignal.setBottomRight(
					QPoint(ptree.get<int>("signal.bottom_right_x", 0),
						ptree.get<int>("signal.bottom_right_y", 0)));

				rectNoise.setTopLeft(
					QPoint(ptree.get<int>("noise.top_left_x", 0),
						ptree.get<int>("noise.top_left_y", 0)));
				rectNoise.setBottomRight(
					QPoint(ptree.get<int>("noise.bottom_right_x", 0),
						ptree.get<int>("noise.bottom_right_y", 0)));

				//gl_widget.set_signal_selection(rectSignal);
				//gl_widget.set_noise_selection(rectNoise);
				//gl_widget.enable_selection();
			}
			catch (std::exception& e)
			{
				display_error("Couldn't load ini file\n" + std::string(e.what()));
			}
		}

		void MainWindow::load_convo_matrix()
		{
			QLineEdit* path_line_edit = findChild<QLineEdit*>("ConvoMatrixPathLineEdit");
			const std::string path = path_line_edit->text().toUtf8();
			boost::property_tree::ptree ptree;
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			std::stringstream strStream;
			std::string str;
			std::string delims = " \f\n\r\t\v";
			std::vector<std::string> v_str, matrix_size, matrix;
			set_convolution_mode(false);
			holovibes_.reset_convolution_matrix();

			try
			{
				std::ifstream file(path);
				uint c = 0;

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
				cd.convo_matrix_width = std::stoi(matrix_size[0]);
				cd.convo_matrix_height = std::stoi(matrix_size[1]);
				cd.convo_matrix_z = std::stoi(matrix_size[2]);
				boost::trim(v_str[1]);
				boost::split(matrix, v_str[1], boost::is_any_of(delims), boost::token_compress_on);
				while (c < matrix.size())
				{
					if (matrix[c] != "")
						cd.convo_matrix.push_back(std::stof(matrix[c]));
					c++;
				}
				if ((cd.convo_matrix_width * cd.convo_matrix_height * cd.convo_matrix_z) != matrix.size())
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

			QLineEdit* path_line_edit = findChild<QLineEdit *>("ImageOutputPathLineEdit");
			path_line_edit->clear();
			path_line_edit->insert(filename);
		}

		std::string MainWindow::set_record_filename_properties(FrameDescriptor fd, std::string filename)
		{
			std::string mode = (is_direct_mode() ? "D" : "H");
			size_t i;

			std::string sub_str = "_" + mode
				+ "_" + std::to_string(fd.width)
				+ "_" + std::to_string(fd.height)
				+ "_" + std::to_string(static_cast<int>(fd.depth) << 3) + "bit"
				+ "_" + "e"; // Holovibes record only in little endian

			for (i = filename.length(); i >= 0; --i)
				if (filename[i] == '.')
					break;

			if (i != 0)
				filename.insert(i, sub_str, 0, sub_str.length());
			return (filename);
		}

		void MainWindow::set_record()
		{
			QSpinBox*  nb_of_frames_spinbox = findChild<QSpinBox*>("NumberOfFramesSpinBox");
			QLineEdit* path_line_edit = findChild<QLineEdit*>("ImageOutputPathLineEdit");
			QCheckBox* float_output_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");
			QCheckBox* complex_output_checkbox = findChild<QCheckBox*>("RecordComplexOutputCheckBox");

			int nb_of_frames = nb_of_frames_spinbox->value();
			std::string path = path_line_edit->text().toUtf8();
			if (path == "")
				return;
			Queue* queue;

			try
			{
				if (float_output_checkbox->isChecked() && !is_direct_mode())
				{
					std::shared_ptr<ICompute> pipe = holovibes_.get_pipe();
					FrameDescriptor frame_desc = holovibes_.get_output_queue().get_frame_desc();

					frame_desc.depth = sizeof(float);
					queue = new Queue(frame_desc, global::global_config.float_queue_max_size, "FloatQueue");
					pipe->request_float_output(queue);
				}
				else if (complex_output_checkbox->isChecked() && !is_direct_mode())
				{
					std::shared_ptr<ICompute> pipe = holovibes_.get_pipe();
					FrameDescriptor frame_desc = holovibes_.get_output_queue().get_frame_desc();

					frame_desc.depth = sizeof(cufftComplex);
					queue = new Queue(frame_desc, global::global_config.float_queue_max_size, "ComplexQueue");
					pipe->request_complex_output(queue);
				}
				else if (is_direct_mode())
					queue = &holovibes_.get_capture_queue();
				else
					queue = &holovibes_.get_output_queue();

				path = set_record_filename_properties(queue->get_frame_desc(), path);
				record_thread_.reset(new ThreadRecorder(
					*queue,
					path,
					nb_of_frames,
					this));

				connect(record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_image_record()));
				record_thread_->start();

				QPushButton* cancel_button = findChild<QPushButton *>("ImageOutputStopPushButton");
				cancel_button->setDisabled(false);
			}
			catch (std::exception& e)
			{
				display_error(e.what());
			}
		}

		void MainWindow::finished_image_record()
		{
			QCheckBox* float_output_checkbox = findChild<QCheckBox *>("RecordFloatOutputCheckBox");
			QCheckBox* complex_output_checkbox = findChild<QCheckBox *>("RecordComplexOutputCheckBox");
			QProgressBar*   progress_bar = InfoManager::get_manager()->get_progress_bar();

			record_thread_.reset(nullptr);

			progress_bar->setMaximum(1);
			progress_bar->setValue(1);
			if (float_output_checkbox->isChecked() && !is_direct_mode())
				holovibes_.get_pipe()->request_float_output_stop();
			if (complex_output_checkbox->isChecked() && !is_direct_mode())
				holovibes_.get_pipe()->request_complex_output_stop();
			display_info("Record done");
		}

		void MainWindow::average_record()
		{
			if (plot_window_)
			{
				plot_window_->stop_drawing();
				plot_window_.reset(nullptr);
				pipe_refresh();
			}

			QSpinBox* nb_of_frames_spin_box = findChild<QSpinBox*>("NumberOfFramesSpinBox");
			nb_frames_ = nb_of_frames_spin_box->value();
			QLineEdit* output_line_edit = findChild<QLineEdit*>("ROIOutputPathLineEdit");
			std::string output_path = output_line_edit->text().toUtf8();

			CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
				holovibes_.get_average_queue(),
				output_path,
				nb_frames_,
				this));
			connect(CSV_record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_average_record()));
			CSV_record_thread_->start();

			QPushButton* roi_stop_push_button = findChild<QPushButton*>("ROIOuputStopPushButton");
			roi_stop_push_button->setDisabled(false);
		}

		void MainWindow::finished_average_record()
		{
			CSV_record_thread_.reset(nullptr);
			display_info("ROI record done");

			QPushButton* roi_stop_push_button = findChild<QPushButton*>("ROIOuputStopPushButton");
			roi_stop_push_button->setDisabled(true);
		}

		void MainWindow::browse_batch_input()
		{
			QString filename = QFileDialog::getOpenFileName(this,
				tr("Batch input file"), "C://", tr("All files (*)"));

			QLineEdit* batch_input_line_edit = findChild<QLineEdit*>("BatchInputPathLineEdit");
			batch_input_line_edit->clear();
			batch_input_line_edit->insert(filename);
		}

		void MainWindow::image_batch_record()
		{
			QLineEdit* output_path = findChild<QLineEdit*>("ImageOutputPathLineEdit");

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
				pipe_refresh();
			}

			QLineEdit* output_path = findChild<QLineEdit*>("ROIOutputPathLineEdit");

			is_batch_img_ = false;
			is_batch_interrupted_ = false;
			batch_record(std::string(output_path->text().toUtf8()));
		}

		void MainWindow::batch_record(const std::string& path)
		{
			file_index_ = 1;
			QLineEdit* batch_input_line_edit = findChild<QLineEdit*>("BatchInputPathLineEdit");
			QSpinBox * frame_nb_spin_box = findChild<QSpinBox*>("NumberOfFramesSpinBox");

			// Getting the path to the input batch file, and the number of frames to record.
			const std::string input_path = batch_input_line_edit->text().toUtf8();
			const uint frame_nb = frame_nb_spin_box->value();

			try
			{
				Queue* q;
				if (is_direct_mode())
					q = &holovibes_.get_capture_queue();
				else
					q = &holovibes_.get_output_queue();
				// Only loading the dll at runtime
				gpib_interface_ = gpib::GpibDLL::load_gpib("gpib.dll", input_path);

				std::string formatted_path = set_record_filename_properties(q->get_frame_desc(), formatted_path);
				formatted_path = format_batch_output(path, file_index_);

				is_enabled_camera_ = false;

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

			QSpinBox * frame_nb_spin_box = findChild<QSpinBox*>("NumberOfFramesSpinBox");
			std::string path;

			if (is_batch_img_)
				path = findChild<QLineEdit*>("ImageOutputPathLineEdit")->text().toUtf8();
			else
				path = findChild<QLineEdit*>("ROIOutputPathLineEdit")->text().toUtf8();

			Queue* q;
			if (is_direct_mode())
				q = &holovibes_.get_capture_queue();
			else
				q = &holovibes_.get_output_queue();

			std::string output_filename = format_batch_output(path, file_index_);
			const uint frame_nb = frame_nb_spin_box->value();
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
			is_enabled_camera_ = true;
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
			static QString tmp_path = "";
			QString filename = "";

			filename = QFileDialog::getOpenFileName(this,
				tr("import file"), ((tmp_path == "") ? ("C://") : (tmp_path)), tr("All files (*)"));

			QLineEdit* import_line_edit = findChild<QLineEdit*>("ImportPathLineEdit");

			if (filename != "")
			{
				import_line_edit->clear();
				import_line_edit->insert(filename);
				tmp_path = filename;
			}
		}

		void MainWindow::import_file_stop(void)
		{
			close_critical_compute();
			camera_none();
			close_windows();
			remove_infos();
			holovibes_.get_compute_desc().compute_mode.exchange(Computation::Stop);
			notify();
		}

		void MainWindow::import_file()
		{
			import_file_stop();
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			QLineEdit *import_line_edit = findChild<QLineEdit *>("ImportPathLineEdit");
			QSpinBox *width_spinbox = findChild<QSpinBox *>("ImportWidthSpinBox");
			QSpinBox *height_spinbox = findChild<QSpinBox *>("ImportHeightSpinBox");
			QSpinBox *fps_spinbox = findChild<QSpinBox *>("ImportFpsSpinBox");
			QSpinBox *start_spinbox = findChild<QSpinBox *>("ImportStartSpinBox");
			QSpinBox *end_spinbox = findChild<QSpinBox *>("ImportEndSpinBox");
			QComboBox *depth_spinbox = findChild<QComboBox *>("ImportDepthComboBox");
			QComboBox *big_endian_checkbox = findChild<QComboBox *>("ImportEndiannessComboBox");
			QCheckBox *cine = findChild<QCheckBox *>("CineFileCheckBox");
			cd.stft_steps.exchange(std::ceil(static_cast<float>(fps_spinbox->value()) / 20.0f));
			int	depth_multi = 1;
			std::string file_src = import_line_edit->text().toUtf8();

			try
			{
				if (cine->isChecked() == true)
					seek_cine_header_data(file_src, holovibes_);
				if (file_src == "")
					throw std::exception("[IMPORT] No input file");
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
			FrameDescriptor frame_desc = {
				static_cast<ushort>(width_spinbox->value()),
				static_cast<ushort>(height_spinbox->value()),
				static_cast<float>(depth_multi),
				static_cast<float>(cd.import_pixel_size.load()),
				(big_endian_checkbox->currentText() == QString("Big Endian") ?
					endianness::BIG_ENDIAN : endianness::LITTLE_ENDIAN) };
			is_enabled_camera_ = false;
			mainDisplay.reset(nullptr);
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
				is_enabled_camera_ = false;
				mainDisplay.reset(nullptr);
				holovibes_.dispose_compute();
				holovibes_.dispose_capture();
				return;
			}
			is_enabled_camera_ = true;
			set_image_mode();

			// Changing the gui
			findChild<QLineEdit *>("BoundaryLineEdit")->clear();
			findChild<QLineEdit *>("BoundaryLineEdit")->insert(QString::number(holovibes_.get_boundary()));
			if (depth_spinbox->currentText() == QString("16") && cine->isChecked() == false)
				big_endian_checkbox->setEnabled(true);
			QAction *settings = findChild<QAction*>("actionSettings");
			settings->setEnabled(false);
			if (holovibes_.get_tcapture()->stop_requested_)
			{
				is_enabled_camera_ = false;
				mainDisplay.reset(nullptr);
				holovibes_.dispose_compute();
				holovibes_.dispose_capture();
			}
			notify();
		}

		void MainWindow::import_start_spinbox_update()
		{
			QSpinBox *start_spinbox = findChild<QSpinBox *>("ImportStartSpinBox");
			QSpinBox *end_spinbox = findChild<QSpinBox *>("ImportEndSpinBox");

			if (start_spinbox->value() > end_spinbox->value())
				end_spinbox->setValue(start_spinbox->value());
		}

		void MainWindow::import_end_spinbox_update()
		{
			QSpinBox *start_spinbox = findChild<QSpinBox *>("ImportStartSpinBox");
			QSpinBox *end_spinbox = findChild<QSpinBox *>("ImportEndSpinBox");

			if (end_spinbox->value() < start_spinbox->value())
				start_spinbox->setValue(end_spinbox->value());
		}

		void MainWindow::closeEvent(QCloseEvent* event)
		{
			close_critical_compute();
			camera_none();
			close_windows();
			remove_infos();
			// Avoiding "unused variable" warning.
			static_cast<void*>(event);
			save_ini("holovibes.ini");
		}

		void MainWindow::change_camera(const Holovibes::camera_type camera_type)
		{
			close_critical_compute();
			close_windows();
			remove_infos();
			if (camera_type != Holovibes::NONE)
			{
				try
				{
					mainDisplay.reset(nullptr);
					if (!is_direct_mode())
						holovibes_.dispose_compute();
					holovibes_.dispose_capture();
					holovibes_.init_capture(camera_type);
					set_image_mode();
					camera_type_ = camera_type;
					QAction* settings = findChild<QAction*>("actionSettings");
					settings->setEnabled(true);
					notify();
				}
				catch (CameraException& e)
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
			InfoManager::get_manager()->remove_info("Error");
			InfoManager::get_manager()->update_info("Error", msg);
		}

		void MainWindow::display_info(const std::string msg)
		{
			InfoManager::get_manager()->remove_info("Info");
			InfoManager::get_manager()->update_info("Info", msg);
		}

		void MainWindow::open_file(const std::string& path)
		{
			QDesktopServices::openUrl(QUrl::fromLocalFile(QString(path.c_str())));
		}

		void MainWindow::load_ini(const std::string& path)
		{
			boost::property_tree::ptree ptree;
			GroupBox *image_rendering_group_box = findChild<GroupBox *>("ImageRenderingGroupBox");
			GroupBox *view_group_box = findChild<GroupBox *>("ViewGroupBox");
			GroupBox *special_group_box = findChild<GroupBox *>("PostProcessingGroupBox");
			GroupBox *record_group_box = findChild<GroupBox *>("RecordGroupBox");
			GroupBox *import_group_box = findChild<GroupBox *>("ImportGroupBox");
			GroupBox *info_group_box = findChild<GroupBox *>("InfoGroupBox");

			QAction*	image_rendering_action = findChild<QAction*>("actionImage_rendering");
			QAction*	view_action = findChild<QAction*>("actionView");
			QAction*	special_action = findChild<QAction*>("actionSpecial");
			QAction*	record_action = findChild<QAction*>("actionRecord");
			QAction*	import_action = findChild<QAction*>("actionImport");
			QAction*	info_action = findChild<QAction*>("actionInfo");

			try
			{
				boost::property_tree::ini_parser::read_ini(path, ptree);
			}
			catch (std::exception& e)
			{
				std::cout << e.what() << std::endl;
			}

			ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (!ptree.empty())
			{
				Config& config = global::global_config;
				// Config
				config.input_queue_max_size = ptree.get<int>("config.input_buffer_size", config.input_queue_max_size);
				config.output_queue_max_size = ptree.get<int>("config.output_buffer_size", config.output_queue_max_size);
				config.float_queue_max_size = ptree.get<int>("config.float_buffer_size", config.float_queue_max_size);
				config.frame_timeout = ptree.get<int>("config.frame_timeout", config.frame_timeout);
				config.flush_on_refresh = ptree.get<int>("config.flush_on_refresh", config.flush_on_refresh);
				config.reader_buf_max_size = ptree.get<int>("config.input_file_buffer_size", config.reader_buf_max_size);
				cd.special_buffer_size.exchange(ptree.get<int>("config.convolution_buffer_size", cd.special_buffer_size.load()));
				cd.stft_level.exchange(ptree.get<uint>("config.stft_buffer_size", cd.stft_level.load()));
				cd.ref_diff_level.exchange(ptree.get<uint>("config.reference_buffer_size", cd.ref_diff_level.load()));
				cd.img_acc_level.exchange(ptree.get<uint>("config.accumulation_buffer_size", cd.img_acc_level.load()));

				// Camera type
				const int camera_type = ptree.get<int>("image_rendering.camera", 0);
				change_camera((Holovibes::camera_type)camera_type);

				// Image rendering
				image_rendering_action->setChecked(!ptree.get<bool>("image_rendering.hidden", false));
				image_rendering_group_box->setHidden(ptree.get<bool>("image_rendering.hidden", false));

				const ushort p_nsample = ptree.get<ushort>("image_rendering.phase_number", cd.nsamples.load());
				cd.nsamples.exchange(1);
				/*if (p_nsample < 1)
					cd.nsamples.exchange(1);
				else if (p_nsample > config.input_queue_max_size)
					cd.nsamples.exchange(config.input_queue_max_size);
				else
					cd.nsamples.exchange(p_nsample);*/

				const ushort p_index = ptree.get<ushort>("image_rendering.p_index", cd.pindex.load());
				if (p_index >= 0 && p_index < cd.nsamples.load())
					cd.pindex.exchange(p_index);

				cd.lambda.exchange(ptree.get<float>("image_rendering.lambda", cd.lambda.load()));

				cd.zdistance.exchange(ptree.get<float>("image_rendering.z_distance", cd.zdistance.load()));

				const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
				if (z_step > 0.0f)
					z_step_ = z_step;

				cd.algorithm.exchange(static_cast<Algorithm>(
					ptree.get<int>("image_rendering.algorithm", cd.algorithm.load())));

				// View
				view_action->setChecked(!ptree.get<bool>("view.hidden", false));
				view_group_box->setHidden(ptree.get<bool>("view.hidden", false));

				cd.view_mode.exchange(static_cast<ComplexViewMode>(
					ptree.get<int>("view.view_mode", cd.view_mode.load())));

				cd.log_scale_enabled.exchange(
					ptree.get<bool>("view.log_scale_enabled", cd.log_scale_enabled.load()));

				cd.shift_corners_enabled.exchange(
					ptree.get<bool>("view.shift_corners_enabled", cd.shift_corners_enabled.load()));

				cd.contrast_enabled.exchange(
					ptree.get<bool>("view.contrast_enabled", cd.contrast_enabled.load()));

				cd.contrast_min.exchange(ptree.get<float>("view.contrast_min", cd.contrast_min.load()));

				cd.contrast_max.exchange(ptree.get<float>("view.contrast_max", cd.contrast_max.load()));

				cd.img_acc_enabled.exchange(ptree.get<bool>("view.accumulation_enabled", cd.img_acc_enabled.load()));
				//main_rotate = ptree.get("view.mainWindow_rotate", main_rotate/* / 90*/); //TODO
				xzAngle = ptree.get<float>("view.xCut_rotate", xzAngle);
				yzAngle = ptree.get<float>("view.yCut_rotate", yzAngle);
				//mainflip = ptree.get("view.mainWindow_flip", mainflip); //TODO
				xzFlip = ptree.get("view.xCut_flip", xzFlip);
				yzFlip = ptree.get("view.yCut_flip", yzFlip);

				// Post Processing
				special_action->setChecked(!ptree.get<bool>("post_processing.hidden", false));
				special_group_box->setHidden(ptree.get<bool>("post_processing.hidden", false));
				cd.vibrometry_q.exchange(
					ptree.get<int>("post_processing.image_ratio_q", cd.vibrometry_q.load()));
				is_enabled_average_ = ptree.get<bool>("post_processing.average_enabled", is_enabled_average_);
				cd.average_enabled.exchange(is_enabled_average_);

				// Record
				record_action->setChecked(!ptree.get<bool>("record.hidden", false));
				record_group_box->setHidden(ptree.get<bool>("record.hidden", false));

				// Import
				import_action->setChecked(!ptree.get<bool>("import.hidden", false));
				import_group_box->setHidden(ptree.get<bool>("import.hidden", false));
				config.import_pixel_size = ptree.get<float>("import.pixel_size", config.import_pixel_size);
				cd.import_pixel_size.exchange(config.import_pixel_size);

				// Info
				info_action->setChecked(!ptree.get<bool>("info.hidden", false));
				info_group_box->setHidden(ptree.get<bool>("info.hidden", false));
				theme_index_ = ptree.get<int>("info.theme_type", theme_index_);

				// Autofocus
				cd.autofocus_size.exchange(ptree.get<int>("autofocus.size", cd.autofocus_size.load()));
				cd.autofocus_z_min.exchange(ptree.get<float>("autofocus.z_min", cd.autofocus_z_min.load()));
				cd.autofocus_z_max.exchange(ptree.get<float>("autofocus.z_max", cd.autofocus_z_max.load()));
				cd.autofocus_z_div.exchange(ptree.get<uint>("autofocus.steps", cd.autofocus_z_div.load()));
				cd.autofocus_z_iter.exchange(ptree.get<uint>("autofocus.loops", cd.autofocus_z_iter.load()));

				//flowgraphy
				uint flowgraphy_level = ptree.get<uint>("flowgraphy.level", cd.flowgraphy_level.load());
				if (flowgraphy_level % 2 == 0)
					flowgraphy_level++;
				cd.flowgraphy_level.exchange(flowgraphy_level);
				cd.flowgraphy_enabled.exchange(ptree.get<bool>("flowgraphy.enable", cd.flowgraphy_enabled.load()));

				// Reset button
				config.set_cuda_device = ptree.get<bool>("reset.set_cuda_device", config.set_cuda_device);
				config.auto_device_number = ptree.get<bool>("reset.auto_device_number", config.auto_device_number);
				config.device_number = ptree.get<int>("reset.device_number", config.device_number);

			}
		}

		void MainWindow::save_ini(const std::string& path)
		{
			///import_file_stop(); // Tmp
			boost::property_tree::ptree ptree;
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			if (cd.stft_enabled.load())
			{
				cd.nsamples.exchange(cd.stft_level.load());
				cd.stft_level.exchange(cd.nsamples.load());
			}
			GroupBox *image_rendering_group_box = findChild<GroupBox *>("ImageRenderingGroupBox");
			GroupBox *view_group_box = findChild<GroupBox *>("ViewGroupBox");
			GroupBox *special_group_box = findChild<GroupBox *>("PostProcessingGroupBox");
			GroupBox *record_group_box = findChild<GroupBox *>("RecordGroupBox");
			GroupBox *import_group_box = findChild<GroupBox *>("ImportGroupBox");
			GroupBox *info_group_box = findChild<GroupBox *>("InfoGroupBox");
			Config& config = global::global_config;

			// Config
			ptree.put("config.input_buffer_size", config.input_queue_max_size);
			ptree.put("config.output_buffer_size", config.output_queue_max_size);
			ptree.put("config.float_buffer_size", config.float_queue_max_size);
			ptree.put("config.input_file_buffer_size", config.reader_buf_max_size);
			ptree.put("config.stft_buffer_size", cd.stft_level.load());
			ptree.put("config.reference_buffer_size", cd.ref_diff_level.load());
			ptree.put("config.accumulation_buffer_size", cd.img_acc_level.load());
			ptree.put("config.convolution_buffer_size", cd.special_buffer_size.load());
			ptree.put("config.frame_timeout", config.frame_timeout);
			ptree.put<bool>("config.flush_on_refresh", config.flush_on_refresh);

			// Image rendering
			ptree.put<bool>("image_rendering.hidden", image_rendering_group_box->isHidden());
			ptree.put("image_rendering.camera", camera_type_);
			ptree.put("image_rendering.phase_number", cd.nsamples.load());
			ptree.put("image_rendering.p_index", cd.pindex.load());
			ptree.put("image_rendering.lambda", cd.lambda.load());
			ptree.put("image_rendering.z_distance", cd.zdistance.load());
			ptree.put("image_rendering.z_step", z_step_);
			ptree.put("image_rendering.algorithm", cd.algorithm.load());

			// View
			ptree.put<bool>("view.hidden", view_group_box->isHidden());
			ptree.put("view.view_mode", cd.view_mode.load());
			ptree.put<bool>("view.log_scale_enabled", cd.log_scale_enabled.load());
			ptree.put<bool>("view.shift_corners_enabled", cd.shift_corners_enabled.load());
			ptree.put<bool>("view.contrast_enabled", cd.contrast_enabled.load());
			ptree.put("view.contrast_min", cd.contrast_min.load());
			ptree.put("view.contrast_max", cd.contrast_max.load());
			ptree.put<bool>("view.accumulation_enabled", cd.img_acc_enabled.load());
			//ptree.put("view.mainWindow_rotate", main_rotate); //TODO
			ptree.put<float>("view.xCut_rotate", xzAngle);
			ptree.put<float>("view.yCut_rotate", yzAngle);
			//ptree.put("view.mainWindow_flip", mainflip); //TODO
			ptree.put("view.xCut_flip", xzFlip);
			ptree.put("view.yCut_flip", yzFlip);

			// Post-processing
			ptree.put<bool>("post_processing.hidden", special_group_box->isHidden());
			ptree.put("post_processing.image_ratio_q", cd.vibrometry_q.load());
			ptree.put<bool>("post_processing.average_enabled", is_enabled_average_);

			// Record
			ptree.put<bool>("record.hidden", record_group_box->isHidden());

			// Import
			ptree.put<bool>("import.hidden", import_group_box->isHidden());
			ptree.put("import.pixel_size", cd.import_pixel_size.load());

			// Info
			ptree.put<bool>("info.hidden", info_group_box->isHidden());
			ptree.put("info.theme_type", theme_index_);

			// Autofocus
			ptree.put("autofocus.size", cd.autofocus_size.load());
			ptree.put("autofocus.z_min", cd.autofocus_z_min.load());
			ptree.put("autofocus.z_max", cd.autofocus_z_max.load());
			ptree.put("autofocus.steps", cd.autofocus_z_div.load());
			ptree.put("autofocus.loops", cd.autofocus_z_iter.load());

			//flowgraphy
			ptree.put("flowgraphy.level", cd.flowgraphy_level.load());
			ptree.put<bool>("flowgraphy.enable", cd.flowgraphy_enabled.load());

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

		std::string MainWindow::format_batch_output(const std::string& path, const uint index)
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
			QComboBox* depth_cbox = findChild<QComboBox*>("ImportDepthComboBox");
			QString curr_value = depth_cbox->currentText();
			QComboBox* imp_cbox = findChild<QComboBox*>("ImportEndiannessComboBox");

			// Changing the endianess when depth = 8 makes no sense
			imp_cbox->setEnabled(curr_value == "16");
		}

		void MainWindow::change_window()
		{
			QComboBox *window_cbox = findChild<QComboBox*>("WindowSelectionComboBox");
			ComputeDescriptor& cd = holovibes_.get_compute_desc();

			if (window_cbox->currentIndex() == 0)
				cd.current_window.exchange(WindowKind::MainDisplay);
			else if (window_cbox->currentIndex() == 1)
				cd.current_window.exchange(WindowKind::SliceXZ);
			else if (window_cbox->currentIndex() == 2)
				cd.current_window.exchange(WindowKind::SliceYZ);
			notify();
		}

		void MainWindow::set_import_cine_file(bool value)
		{
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			cd.is_cine_file.exchange(value);
			notify();
		}

		void MainWindow::seek_cine_header_data(std::string &file_src_, Holovibes& holovibes_)
		{
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			QComboBox		*depth_spinbox = findChild<QComboBox*>("ImportDepthComboBox");
			int				read_width = 0, read_height = 0;
			ushort			read_depth = 0;
			uint			read_pixelpermeter_x = 0, offset_to_ptr = 0;
			FILE*			file = nullptr;
			fpos_t			pos = 0;
			size_t			length = 0;
			char			buffer[44];
			double			pixel_size = 0;

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
				if ((length = std::fread(&read_depth, 1, sizeof(short), file)) = !sizeof(short))
					throw std::runtime_error("[READER] unable to read file: " + file_src_);
				/*Reading value biXpelsPerMetter*/
				pos = offset_to_ptr + 24;
				std::fsetpos(file, &pos);
				if ((length = std::fread(&read_pixelpermeter_x, 1, sizeof(int), file)) = !sizeof(int))
					throw std::runtime_error("[READER] unable to read file: " + file_src_);

				/*Setting value in Qt interface*/
				depth_spinbox->setCurrentIndex((read_depth != 8));

				findChild<QSpinBox*>("ImportWidthSpinBox")->setValue(read_width);
				if (read_height < 0)
					read_height = -read_height;
				findChild<QSpinBox*>("ImportHeightSpinBox")->setValue(read_height);
				pixel_size = (1 / static_cast<double>(read_pixelpermeter_x)) * 1000000;
				cd.import_pixel_size.exchange(pixel_size);
				findChild<QComboBox*>("ImportEndiannessComboBox")->setCurrentIndex(0); // Little Endian

				/*Unused fonction ready to read framerate in exposure*/
				//get_framerate_cinefile(file, file_src_);
				//get_exposure_cinefile(file, file_src_);
				notify();
			}
			catch (std::runtime_error& e)
			{
				std::cout << e.what() << std::endl;
				throw std::runtime_error(e.what());
			}
		}

		void MainWindow::cancel_stft_view(ComputeDescriptor& cd)
		{
			if (cd.signal_trig_enabled.load())
				stft_signal_trig(false);
			else if (cd.stft_view_enabled.load())
				cancel_stft_slice_view();
			cd.stft_view_enabled.exchange(false);
			cd.stft_enabled.exchange(false);
			cd.signal_trig_enabled.exchange(false);
			cd.nsamples.exchange(1);
			notify();
		}

		void MainWindow::close_critical_compute()
		{
			ComputeDescriptor& cd = holovibes_.get_compute_desc();
			if (cd.average_enabled.load())
				set_average_mode(false);
			if (cd.stft_enabled.load())
				cancel_stft_view(cd);
			if (cd.ref_diff_enabled.load() || cd.ref_sliding_enabled.load())
				cancel_take_reference();
			if (cd.filter_2d_enabled.load())
				cancel_filter2D();
			holovibes_.dispose_compute();
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

		void MainWindow::stft_signal_trig(bool checked)
		{
			QCheckBox* stft = findChild<QCheckBox*>("STFTCheckBox");
			QCheckBox* stft_view = findChild<QCheckBox*>("STFTCutsCheckBox");
			QCheckBox* trig = findChild<QCheckBox*>("ExtTrigCheckBox");
			ComputeDescriptor&	cd = holovibes_.get_compute_desc();

			if (checked)
			{
				if (cd.stft_view_enabled.load())
					cancel_stft_slice_view();

				// TODO add a wait for a external signal to trigger STFT at the next step
				set_stft(false);
				set_stft(true);

				cd.signal_trig_enabled.exchange(true);
			}
			else
				cd.signal_trig_enabled.exchange(false);
			notify();
		}

		void MainWindow::title_detect(void)
		{
			QLineEdit	*import_line_edit = findChild<QLineEdit*>("ImportPathLineEdit");
			QSpinBox	*import_width_box = findChild<QSpinBox*>("ImportWidthSpinBox");
			QSpinBox	*import_height_box = findChild<QSpinBox*>("ImportHeightSpinBox");
			QComboBox	*import_depth_box = findChild<QComboBox*>("ImportDepthComboBox");
			QComboBox	*import_endian_box = findChild<QComboBox*>("ImportEndiannessComboBox");
			const std::string	file_src = import_line_edit->text().toUtf8();
			uint				width = 0, height = 0, depth = 0, underscore = 5;
			size_t				i;
			bool				mode, endian;

			for (i = file_src.length(); i > 0 && underscore; --i)
				if (file_src[i] == '_')
					underscore--;
			if (underscore)
				return (display_error("Cannot detect title properties"));
			if (file_src[++i] == '_' && i++)
				if (file_src[i] == 'D' || file_src[i] == 'H')
					mode = ((file_src[i] == 'D') ? (false) : (true));
				else
					return (display_error("Cannot detect title properties"));
			if (file_src[++i] == '_')
			{
				width = std::atoi(&file_src[++i]);
				while (file_src[i] != '_' && file_src[i])
					++i;
			}
			else
				return (display_error("Cannot detect title properties"));
			if (file_src[i++] == '_')
			{
				height = std::atoi(&file_src[i++]);
				while (file_src[i] != '_' && file_src[i])
					++i;
			}
			else
				return (display_error("Cannot detect title properties"));
			if (file_src[i++] == '_')
			{
				depth = std::atoi(&file_src[i++]);
				while (file_src[i] != '_' && file_src[i])
					++i;
			}
			else
				return (display_error("Cannot detect title properties"));
			if (file_src[i++] == '_')
			{
				if (file_src[i] == 'e' || file_src[i] == 'E')
					endian = ((file_src[i] == 'e') ? (false) : (true));
				else
					return (display_error("Cannot detect title properties"));
			}
			if (depth != 8 && depth != 16 && depth != 32 && depth != 64)
				return (display_error("Cannot detect title properties"));
			import_width_box->setValue(width);
			import_height_box->setValue(height);
			import_depth_box->setCurrentIndex(log2(depth) - 3);
			import_endian_box->setCurrentIndex(endian);
		}
	}
}
