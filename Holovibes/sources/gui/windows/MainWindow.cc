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
#include <filesystem>

namespace holovibes
{
	using camera::FrameDescriptor;
	using camera::Endianness;
	namespace gui
	{
		namespace {
			void spinBoxDecimalPointReplacement(QDoubleSpinBox *doubleSpinBox)
			{
				class DoubleValidator : public QValidator
				{
					const QValidator *old;
				public:
					DoubleValidator(const QValidator *old_)
						: QValidator(const_cast<QValidator*>(old_)), old(old_)
					{}

					void fixup(QString & input) const
					{
						input.replace(".", QLocale().decimalPoint());
						input.replace(",", QLocale().decimalPoint());
						old->fixup(input);
					}
					QValidator::State validate(QString & input, int & pos) const
					{
						fixup(input);
						return old->validate(input, pos);
					}
				};
				QLineEdit *lineEdit = doubleSpinBox->findChild<QLineEdit*>();
				lineEdit->setValidator(new DoubleValidator(lineEdit->validator()));
			}
		}
		#pragma region Constructor - Destructor
		MainWindow::MainWindow(Holovibes& holovibes, QWidget *parent)
			: QMainWindow(parent),
			holovibes_(holovibes),
			mainDisplay(nullptr),
			sliceXZ(nullptr),
			sliceYZ(nullptr),
			displayAngle(0.f),
			xzAngle(0.f),
			yzAngle(0.f),
			displayFlip(0),
			xzFlip(0),
			yzFlip(0),
			is_enabled_camera_(false),
			is_enabled_average_(false),
			is_batch_img_(true),
			is_batch_interrupted_(false),
			z_step_(0.01f),
			kCamera(CameraKind::NONE),
			last_img_type_("Magnitude"),
			plot_window_(nullptr),
			record_thread_(nullptr),
			CSV_record_thread_(nullptr),
			file_index_(1),
			theme_index_(0),
			import_type_(ImportType::None),
			compute_desc_(holovibes_.get_compute_desc())
		{
			ui.setupUi(this);

			connect(this, SIGNAL(request_notify()), this, SLOT(on_notify()));
			connect(this, SIGNAL(update_file_reader_index_signal(int)), this, SLOT(update_file_reader_index(int)));



			setWindowIcon(QIcon("Holovibes.ico"));
			InfoManager::get_manager(ui.InfoGroupBox);

			move(QPoint(532, 554));
			show();

			// Hide non default tab
			ui.CompositeGroupBox->setHidden(true);

			ui.actionSpecial->setChecked(false);
			ui.actionRecord->setChecked(false);
			ui.actionInfo->setChecked(false);

			try
			{
				load_ini(GLOBAL_INI_PATH);
			}
			catch (std::exception&)
			{
				std::cout << GLOBAL_INI_PATH << ": Config file not found. Using default values." << std::endl;
			}

			set_night();

			InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::IMG_SOURCE, "ImgSource", "None");

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

			QComboBox *depth_cbox = ui.ImportDepthComboBox;
			connect(depth_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(hide_endianess()));

			QComboBox *window_cbox = ui.WindowSelectionComboBox;
			connect(window_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(change_window()));

			// Display default values
			compute_desc_.compute_mode = Computation::Direct;
			notify();
			compute_desc_.compute_mode = Computation::Stop;
			notify();
			setFocusPolicy(Qt::StrongFocus);

			// spinBox allow ',' and '.' as decimal point
			spinBoxDecimalPointReplacement(ui.WaveLengthDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.ZDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.ZStepDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.PixelSizeDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.ContrastMaxDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.ContrastMinDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.AutofocusZMinDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.AutofocusZMaxDoubleSpinBox);

			ui.FileReaderProgressBar->hide();
			ui.RecordProgressBar;
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
			InfoManager::get_manager()->stop_display();
		}

		
		#pragma endregion
		/* ------------ */
		#pragma region Notify
		void MainWindow::notify()
		{
			// We can't update gui values from a different thread
			// so we pass it to the right on using a signal
			// (This whole notify thing needs to be cleaned up / removed)
			if (QThread::currentThread() != this->thread())
				emit request_notify();
			else
				on_notify();
		}
		void MainWindow::on_notify()
		{
			const bool is_direct = is_direct_mode();
			if (compute_desc_.compute_mode == Computation::Stop)
			{
				ui.ImageRenderingGroupBox->setEnabled(false);
				ui.ViewGroupBox->setEnabled(false);
				ui.MotionFocusGroupBox->setEnabled(false);
				ui.PostProcessingGroupBox->setEnabled(false);
				ui.RecordGroupBox->setEnabled(false);
				ui.ImportGroupBox->setEnabled(true);
				ui.InfoGroupBox->setEnabled(true);
				ui.PixelSizeDoubleSpinBox->setValue(compute_desc_.pixel_size);
				return;
			}
			else if (compute_desc_.compute_mode == Computation::Direct && is_enabled_camera_)
			{
				ui.ImageRenderingGroupBox->setEnabled(true);
				ui.RecordGroupBox->setEnabled(true);
			}
			else if (compute_desc_.compute_mode == Computation::Hologram && is_enabled_camera_)
			{
				ui.ImageRenderingGroupBox->setEnabled(true);
				ui.ViewGroupBox->setEnabled(true);
				ui.PostProcessingGroupBox->setEnabled(true);
				ui.RecordGroupBox->setEnabled(true);
			}
			ui.MotionFocusGroupBox->setEnabled(true);

			ui.ROIOutputPathLineEdit->setEnabled(!is_direct && compute_desc_.average_enabled);
			ui.ROIOutputToolButton->setEnabled(!is_direct && compute_desc_.average_enabled);
			ui.ROIOutputRecPushButton->setEnabled(!is_direct && compute_desc_.average_enabled);
			ui.ROIOutputBatchPushButton->setEnabled(!is_direct && compute_desc_.average_enabled);
			ui.ROIOutputStopPushButton->setEnabled(!is_direct && compute_desc_.average_enabled);
			ui.ROIFileBrowseToolButton->setEnabled(compute_desc_.average_enabled);
			ui.ROIFilePathLineEdit->setEnabled(compute_desc_.average_enabled);
			ui.SaveROIPushButton->setEnabled(compute_desc_.average_enabled);
			ui.LoadROIPushButton->setEnabled(compute_desc_.average_enabled);

			QPushButton* signalBtn = ui.AverageSignalPushButton;
			signalBtn->setEnabled(compute_desc_.average_enabled);
			signalBtn->setStyleSheet((signalBtn->isEnabled() &&
				mainDisplay && mainDisplay->getKindOfOverlay() == KindOfOverlay::Signal) ? "QPushButton {color: #8E66D9;}" : "");

			QPushButton* noiseBtn = ui.AverageNoisePushButton;
			noiseBtn->setEnabled(compute_desc_.average_enabled);
			noiseBtn->setStyleSheet((noiseBtn->isEnabled() &&
				mainDisplay && mainDisplay->getKindOfOverlay() == KindOfOverlay::Noise) ? "QPushButton {color: #00A4AB;}" : "");

			QPushButton* autofocusBtn = ui.AutofocusRunPushButton;
			if (autofocusBtn->isEnabled() && mainDisplay && mainDisplay->getKindOfOverlay() == KindOfOverlay::Autofocus)
			{
				autofocusBtn->setStyleSheet("QPushButton {color: #FFCC00;}");
				autofocusBtn->setText("Cancel Autofocus");
			}
			else
			{
				autofocusBtn->setStyleSheet("");
				autofocusBtn->setText("Run Autofocus");
			}
			ui.AutofocusZMinDoubleSpinBox->setValue(compute_desc_.autofocus_z_min);
			ui.AutofocusZMaxDoubleSpinBox->setValue(compute_desc_.autofocus_z_max);

			ui.PhaseUnwrap2DCheckBox->
				setEnabled(!is_direct && compute_desc_.img_type == ImgType::PhaseIncrease ||
					compute_desc_.img_type == ImgType::Argument);

			ui.STFTCutsCheckBox->setEnabled(!is_direct && !compute_desc_.filter_2d_enabled);
			ui.STFTCutsCheckBox->setChecked(!is_direct && compute_desc_.stft_view_enabled);

			QPushButton *filter_button = ui.Filter2DPushButton;
			filter_button->setEnabled(!is_direct && !compute_desc_.stft_view_enabled
				&& !compute_desc_.filter_2d_enabled && !compute_desc_.stft_view_enabled);
			filter_button->setStyleSheet((!is_direct && compute_desc_.filter_2d_enabled) ? "QPushButton {color: #009FFF;}" : "");
			ui.CancelFilter2DPushButton->setEnabled(!is_direct && compute_desc_.filter_2d_enabled);

			ui.CropStftCheckBox->setEnabled(!is_direct);

			ui.ContrastCheckBox->setChecked(!is_direct && compute_desc_.contrast_enabled);
			ui.LogScaleCheckBox->setChecked(!is_direct && compute_desc_.log_scale_slice_xy_enabled);
			ui.ContrastMinDoubleSpinBox->setEnabled(!is_direct && compute_desc_.contrast_enabled);
			ui.ContrastMaxDoubleSpinBox->setEnabled(!is_direct && compute_desc_.contrast_enabled);
			ui.AutoContrastPushButton->setEnabled(!is_direct && compute_desc_.contrast_enabled);

			QComboBox *window_selection = ui.WindowSelectionComboBox;
			window_selection->setEnabled((compute_desc_.stft_view_enabled));
			window_selection->setCurrentIndex(window_selection->isEnabled() ? compute_desc_.current_window : 0);

			if (compute_desc_.current_window == WindowKind::XYview)
			{
				ui.ContrastMinDoubleSpinBox
					->setValue((compute_desc_.log_scale_slice_xy_enabled) ? compute_desc_.contrast_min_slice_xy.load() : log10(compute_desc_.contrast_min_slice_xy));
				ui.ContrastMaxDoubleSpinBox
					->setValue((compute_desc_.log_scale_slice_xy_enabled) ? compute_desc_.contrast_max_slice_xy.load() : log10(compute_desc_.contrast_max_slice_xy));
				ui.LogScaleCheckBox->setChecked(!is_direct && compute_desc_.log_scale_slice_xy_enabled);
				ui.ImgAccuCheckBox->setChecked(!is_direct && compute_desc_.img_acc_slice_xy_enabled);
				ui.ImgAccuSpinBox->setValue(compute_desc_.img_acc_slice_xy_level);
				ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(displayAngle))).c_str());
				ui.FlipPushButton->setText(("Flip " + std::to_string(displayFlip)).c_str());
			}
			else if (compute_desc_.current_window == WindowKind::XZview)
			{
				ui.ContrastMinDoubleSpinBox
					->setValue((compute_desc_.log_scale_slice_xz_enabled) ? compute_desc_.contrast_min_slice_xz.load() : log10(compute_desc_.contrast_min_slice_xz));
				ui.ContrastMaxDoubleSpinBox
					->setValue((compute_desc_.log_scale_slice_xz_enabled) ? compute_desc_.contrast_max_slice_xz.load() : log10(compute_desc_.contrast_max_slice_xz));
				ui.LogScaleCheckBox->setChecked(!is_direct && compute_desc_.log_scale_slice_xz_enabled);
				ui.ImgAccuCheckBox->setChecked(!is_direct && compute_desc_.img_acc_slice_xz_enabled);
				ui.ImgAccuSpinBox->setValue(compute_desc_.img_acc_slice_xz_level);
				ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(xzAngle))).c_str());
				ui.FlipPushButton->setText(("Flip " + std::to_string(xzFlip)).c_str());
			}
			else if (compute_desc_.current_window == WindowKind::YZview)
			{
				ui.ContrastMinDoubleSpinBox
					->setValue((compute_desc_.log_scale_slice_yz_enabled) ? compute_desc_.contrast_min_slice_yz.load() : log10(compute_desc_.contrast_min_slice_yz));
				ui.ContrastMaxDoubleSpinBox
					->setValue((compute_desc_.log_scale_slice_yz_enabled) ? compute_desc_.contrast_max_slice_yz.load() : log10(compute_desc_.contrast_max_slice_yz));
				ui.LogScaleCheckBox->setChecked(!is_direct && compute_desc_.log_scale_slice_yz_enabled);
				ui.ImgAccuCheckBox->setChecked(!is_direct && compute_desc_.img_acc_slice_yz_enabled);
				ui.ImgAccuSpinBox->setValue(compute_desc_.img_acc_slice_yz_level);
				ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(yzAngle))).c_str());
				ui.FlipPushButton->setText(("Flip " + std::to_string(yzFlip)).c_str());
			}

			ui.FFTShiftCheckBox->setChecked(compute_desc_.shift_corners_enabled);
			ui.PAccuCheckBox->setChecked(compute_desc_.p_accu_enabled);
			ui.PAccSpinBox->setValue(compute_desc_.p_acc_level);

			ui.XAccuCheckBox->setChecked(compute_desc_.x_accu_enabled);
			ui.XAccSpinBox->setValue(compute_desc_.x_acc_level);

			ui.YAccuCheckBox->setChecked(compute_desc_.y_accu_enabled);
			ui.YAccSpinBox->setValue(compute_desc_.y_acc_level);

			QSpinBox *p_vibro = ui.ImageRatioPSpinBox;
			p_vibro->setEnabled(!is_direct && compute_desc_.vibrometry_enabled);
			p_vibro->setValue(compute_desc_.pindex);
			p_vibro->setMaximum(compute_desc_.nsamples - 1);
			QSpinBox *q_vibro = ui.ImageRatioQSpinBox;
			q_vibro->setEnabled(!is_direct && compute_desc_.vibrometry_enabled);
			q_vibro->setValue(compute_desc_.vibrometry_q);
			q_vibro->setMaximum(compute_desc_.nsamples - 1);

			ui.ImageRatioCheckBox->setChecked(!is_direct && compute_desc_.vibrometry_enabled);
			ui.ConvoCheckBox->setEnabled(!is_direct && compute_desc_.convo_matrix.size() != 0);
			ui.AverageCheckBox->setEnabled(!compute_desc_.stft_view_enabled);
			ui.AverageCheckBox->setChecked(!is_direct && compute_desc_.average_enabled);
			ui.FlowgraphyCheckBox->setChecked(!is_direct && compute_desc_.flowgraphy_enabled);
			ui.FlowgraphyLevelSpinBox->setEnabled(!is_direct && compute_desc_.flowgraphy_level);
			ui.FlowgraphyLevelSpinBox->setValue(compute_desc_.flowgraphy_level);
			ui.AutofocusRunPushButton->setEnabled(!is_direct && compute_desc_.algorithm != Algorithm::None && !compute_desc_.stft_view_enabled);
			ui.STFTStepsSpinBox->setEnabled(!is_direct);
			ui.STFTStepsSpinBox->setValue(compute_desc_.stft_steps);
			ui.TakeRefPushButton->setEnabled(!is_direct && !compute_desc_.ref_sliding_enabled);
			ui.SlidingRefPushButton->setEnabled(!is_direct && !compute_desc_.ref_diff_enabled && !compute_desc_.ref_sliding_enabled);
			ui.CancelRefPushButton->setEnabled(!is_direct && (compute_desc_.ref_diff_enabled || compute_desc_.ref_sliding_enabled));
			ui.AlgorithmComboBox->setEnabled(!is_direct);
			ui.AlgorithmComboBox->setCurrentIndex(compute_desc_.algorithm);
			ui.ViewModeComboBox->setCurrentIndex(compute_desc_.img_type);
			ui.PhaseNumberSpinBox->setEnabled(!is_direct && !compute_desc_.stft_view_enabled);
			ui.PhaseNumberSpinBox->setValue(compute_desc_.nsamples);
			ui.PSpinBox->setMaximum(compute_desc_.nsamples - 1);
			ui.PSpinBox->setValue(compute_desc_.pindex);
			ui.WaveLengthDoubleSpinBox->setEnabled(!is_direct);
			ui.WaveLengthDoubleSpinBox->setValue(compute_desc_.lambda * 1.0e9f);
			ui.ZDoubleSpinBox->setEnabled(!is_direct);
			ui.ZDoubleSpinBox->setValue(compute_desc_.zdistance);
			ui.ZStepDoubleSpinBox->setEnabled(!is_direct);

			ui.PixelSizeDoubleSpinBox->setEnabled(!compute_desc_.is_cine_file);
			ui.PixelSizeDoubleSpinBox->setValue(compute_desc_.pixel_size);
			ui.BoundaryLineEdit->setText(QString::number(holovibes_.get_boundary()));
			ui.KernelBufferSizeSpinBox->setValue(compute_desc_.special_buffer_size);
			ui.CineFileCheckBox->setChecked(compute_desc_.is_cine_file);
			ui.ImportWidthSpinBox->setEnabled(!compute_desc_.is_cine_file);
			ui.ImportHeightSpinBox->setEnabled(!compute_desc_.is_cine_file);
			ui.ImportDepthComboBox->setEnabled(!compute_desc_.is_cine_file);
			
			QString depth_value = ui.ImportDepthComboBox->currentText();
			ui.ImportEndiannessComboBox->setEnabled(depth_value == "16" && !compute_desc_.is_cine_file);

			bool isComposite = !is_direct_mode() && compute_desc_.img_type == ImgType::Composite;
			ui.CompositeGroupBox->setHidden(!isComposite);

			// Composite
			QSpinBox *min_box = ui.PMinSpinBox_Composite;
			QSpinBox *max_box = ui.PMaxSpinBox_Composite;
			QDoubleSpinBox *weight_boxes[3];
			weight_boxes[0] = ui.WeightSpinBox_R;
			weight_boxes[1] = ui.WeightSpinBox_G;
			weight_boxes[2] = ui.WeightSpinBox_B;
			Component *components[] = { &compute_desc_.component_r, &compute_desc_.component_g, &compute_desc_.component_b };

			unsigned short pmin = components[0]->p_min;
			unsigned short pmax = components[2]->p_max;

			components[0]->p_min = pmin;
			components[1]->p_min = pmin + (pmax - pmin) * (1/5.f);
			components[0]->p_max = pmin + (pmax - pmin) * (2/5.f);
			components[2]->p_min = pmin + (pmax - pmin) * (3/5.f);
			components[1]->p_max = pmin + (pmax - pmin) * (4/5.f);
			components[2]->p_max = pmax;

			for (int i = 0; i < 3; i++)
				if (components[i]->p_min > components[i]->p_max)
				{
					unsigned short tmp = components[i]->p_min;
					components[i]->p_min = components[i]->p_max.load();
					components[i]->p_max = tmp;
				}

			// We need to store them in a temporary array, otherwise they're erased by the new notify
			float weights[3];
			for (int i = 0; i < 3; i++)
				weights[i] = components[i]->weight;
			for (int i = 0; i < 3; i++)
				weight_boxes[i]->setValue(weights[i]);
			min_box->setMaximum(compute_desc_.nsamples - 1);
			max_box->setMaximum(compute_desc_.nsamples - 1);
			min_box->setValue(pmin);
			max_box->setValue(pmax);
			ui.RenormalizationCheckBox->setChecked(compute_desc_.composite_auto_weights_);

			// Interpolation
			ui.InterpolationCheckbox->setChecked(compute_desc_.interpolation_enabled);
			ui.InterpolationLambda1->setValue(compute_desc_.interp_lambda1 * 1.0e9f);
			ui.InterpolationLambda2->setValue(compute_desc_.interp_lambda2 * 1.0e9f);
			ui.InterpolationSensitivity->setValue(compute_desc_.interp_sensitivity);
			ui.InterpolationShift->setValue(compute_desc_.interp_shift);

			//QCoreApplication::processEvents();
		}

		void MainWindow::notify_error(std::exception& e, const char* msg)
		{
			CustomException* err_ptr = dynamic_cast<CustomException*>(&e);
			std::string str;
			if (err_ptr != nullptr)
			{
				if (err_ptr->get_kind() == error_kind::fail_update)
				{
					// notify will be in close_critical_compute
					compute_desc_.pindex = 0;
					compute_desc_.nsamples = 1;
					if (compute_desc_.flowgraphy_enabled || compute_desc_.convolution_enabled)
					{
						compute_desc_.convolution_enabled = false;
						compute_desc_.flowgraphy_enabled = false;
						compute_desc_.special_buffer_size = 3;
					}
				}
				if (err_ptr->get_kind() == error_kind::fail_accumulation)
				{
					compute_desc_.img_acc_slice_xy_enabled = false;
					compute_desc_.img_acc_slice_xy_level = 1;
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

		void MainWindow::layout_toggled()
		{
			// Resizing to original size, then adjust it to fit the groupboxes
			resize(baseSize());
			adjustSize();
		}

		void MainWindow::display_error(const std::string msg)
		{
			InfoManager::get_manager()->insert_info(InfoManager::InfoType::ERR, "Error", msg);
			InfoManager::get_manager()->startDelError("Error");
		}

		void MainWindow::display_info(const std::string msg)
		{
			InfoManager::get_manager()->insert_info(InfoManager::InfoType::INFO, "Info", msg);
			InfoManager::get_manager()->startDelError("Info");
		}
		
		void MainWindow::credits()
		{
			std::string msg =
				"Holovibes " + version + "\n\n"

				"Developers:\n\n"

				"Eloi Charpentier\n"
				"Julien Gautier\n"
				"Florian Lapeyre\n"

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
		#pragma endregion
		/* ------------ */
		#pragma region Ini
		void MainWindow::configure_holovibes()
		{
			open_file(holovibes_.get_launch_path() + "/" + GLOBAL_INI_PATH);
		}

		void MainWindow::write_ini()
		{
			save_ini(GLOBAL_INI_PATH);
			notify();
		}

		void MainWindow::reload_ini()
		{
			import_file_stop();
			try
			{
				load_ini(GLOBAL_INI_PATH);
			}
			catch (std::exception& e)
			{
				std::cout << e.what() << std::endl;
			}
			if (import_type_ == ImportType::File)
				import_file();
			else if (import_type_ == ImportType::Camera)
				change_camera(kCamera);
			notify();
		}

		void MainWindow::load_ini(const std::string& path)
		{
			boost::property_tree::ptree ptree;
			GroupBox *image_rendering_group_box = ui.ImageRenderingGroupBox;
			GroupBox *view_group_box = ui.ViewGroupBox;
			GroupBox *special_group_box = ui.PostProcessingGroupBox;
			GroupBox *record_group_box = ui.RecordGroupBox;
			GroupBox *import_group_box = ui.ImportGroupBox;
			GroupBox *info_group_box = ui.InfoGroupBox;

			QAction	*image_rendering_action = ui.actionImage_rendering;
			QAction	*view_action = ui.actionView;
			QAction	*special_action = ui.actionSpecial;
			QAction	*record_action = ui.actionRecord;
			QAction	*import_action = ui.actionImport;
			QAction	*info_action = ui.actionInfo;

			boost::property_tree::ini_parser::read_ini(path, ptree);

			if (!ptree.empty())
			{
				Config& config = global::global_config;
				// Config
				config.input_queue_max_size = ptree.get<int>("config.input_buffer_size", config.input_queue_max_size);
				config.output_queue_max_size = ptree.get<int>("config.output_buffer_size", config.output_queue_max_size);
				config.float_queue_max_size = ptree.get<int>("config.float_buffer_size", config.float_queue_max_size);
				config.stft_cuts_output_buffer_size = ptree.get<int>("config.stft_cuts_output_buffer_size", config.stft_cuts_output_buffer_size);
				config.frame_timeout = ptree.get<int>("config.frame_timeout", config.frame_timeout);
				config.flush_on_refresh = ptree.get<int>("config.flush_on_refresh", config.flush_on_refresh);
				config.reader_buf_max_size = ptree.get<int>("config.input_file_buffer_size", config.reader_buf_max_size);
				compute_desc_.special_buffer_size = ptree.get<int>("config.convolution_buffer_size", compute_desc_.special_buffer_size);
				compute_desc_.stft_level = ptree.get<uint>("config.stft_buffer_size", compute_desc_.stft_level);
				compute_desc_.ref_diff_level = ptree.get<uint>("config.reference_buffer_size", compute_desc_.ref_diff_level);
				compute_desc_.img_acc_slice_xy_level = ptree.get<uint>("config.accumulation_buffer_size", compute_desc_.img_acc_slice_xy_level);
				compute_desc_.display_rate = ptree.get<float>("config.display_rate", compute_desc_.display_rate);

				// Camera type
				//const int camera_type = ptree.get<int>("image_rendering.camera", 0);
				//change_camera(static_cast<CameraKind>(camera_type));

				// Image rendering
				image_rendering_action->setChecked(!ptree.get<bool>("image_rendering.hidden", false));
				image_rendering_group_box->setHidden(ptree.get<bool>("image_rendering.hidden", false));

				const ushort p_nsample = ptree.get<ushort>("image_rendering.phase_number", compute_desc_.nsamples);
				if (p_nsample < 1)
					compute_desc_.nsamples = 1;
				else if (p_nsample > config.input_queue_max_size)
					compute_desc_.nsamples = config.input_queue_max_size;
				else
					compute_desc_.nsamples = p_nsample;
				const ushort p_index = ptree.get<ushort>("image_rendering.p_index", compute_desc_.pindex);
				if (p_index >= 0 && p_index < compute_desc_.nsamples)
					compute_desc_.pindex = p_index;

				compute_desc_.lambda = ptree.get<float>("image_rendering.lambda", compute_desc_.lambda);

				compute_desc_.zdistance = ptree.get<float>("image_rendering.z_distance", compute_desc_.zdistance);

				const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
				if (z_step > 0.0f)
					z_step_ = z_step;

				compute_desc_.algorithm = static_cast<Algorithm>(ptree.get<int>("image_rendering.algorithm", compute_desc_.algorithm));

				// View
				view_action->setChecked(!ptree.get<bool>("view.hidden", false));
				view_group_box->setHidden(ptree.get<bool>("view.hidden", false));

				compute_desc_.img_type.exchange(static_cast<ImgType>(
					ptree.get<int>("view.view_mode", compute_desc_.img_type)));
				last_img_type_ = (compute_desc_.img_type == ImgType::Complex) ?
					"Complex output" : (compute_desc_.img_type == ImgType::Composite) ?
						"Composite image" : last_img_type_;

				compute_desc_.log_scale_slice_xy_enabled = ptree.get<bool>("view.log_scale_enabled", compute_desc_.log_scale_slice_xy_enabled);
				compute_desc_.log_scale_slice_xz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_xz", compute_desc_.log_scale_slice_xz_enabled);
				compute_desc_.log_scale_slice_yz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_yz", compute_desc_.log_scale_slice_yz_enabled);

				compute_desc_.shift_corners_enabled = ptree.get<bool>("view.shift_corners_enabled", compute_desc_.shift_corners_enabled);

				compute_desc_.contrast_enabled = ptree.get<bool>("view.contrast_enabled", compute_desc_.contrast_enabled);

				compute_desc_.contrast_min_slice_xy = ptree.get<float>("view.contrast_min", compute_desc_.contrast_min_slice_xy);
				compute_desc_.contrast_max_slice_xy = ptree.get<float>("view.contrast_max", compute_desc_.contrast_max_slice_xy);
				compute_desc_.cuts_contrast_p_offset = ptree.get<ushort>("view.cuts_contrast_p_offset", compute_desc_.cuts_contrast_p_offset);
				if (compute_desc_.cuts_contrast_p_offset < 0)
					compute_desc_.cuts_contrast_p_offset = 0;
				else if (compute_desc_.cuts_contrast_p_offset > compute_desc_.nsamples - 1)
					compute_desc_.cuts_contrast_p_offset = compute_desc_.nsamples - 1;

				compute_desc_.img_acc_slice_xy_enabled = ptree.get<bool>("view.accumulation_enabled", compute_desc_.img_acc_slice_xy_enabled);

				displayAngle = ptree.get("view.mainWindow_rotate", displayAngle);
				xzAngle = ptree.get<float>("view.xCut_rotate", xzAngle);
				yzAngle = ptree.get<float>("view.yCut_rotate", yzAngle);
				displayFlip = ptree.get("view.mainWindow_flip", displayFlip);
				xzFlip = ptree.get("view.xCut_flip", xzFlip);
				yzFlip = ptree.get("view.yCut_flip", yzFlip);

				// Post Processing
				special_action->setChecked(!ptree.get<bool>("post_processing.hidden", false));
				special_group_box->setHidden(ptree.get<bool>("post_processing.hidden", false));
				compute_desc_.vibrometry_q.exchange(
					ptree.get<int>("post_processing.image_ratio_q", compute_desc_.vibrometry_q));
				is_enabled_average_ = ptree.get<bool>("post_processing.average_enabled", is_enabled_average_);
				compute_desc_.average_enabled = is_enabled_average_;

				// Record
				record_action->setChecked(!ptree.get<bool>("record.hidden", false));
				record_group_box->setHidden(ptree.get<bool>("record.hidden", false));

				// Import
				import_action->setChecked(!ptree.get<bool>("import.hidden", false));
				import_group_box->setHidden(ptree.get<bool>("import.hidden", false));
				compute_desc_.pixel_size = ptree.get<float>("import.pixel_size", compute_desc_.pixel_size);
				ui.ImportFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));

				// Info
				info_action->setChecked(!ptree.get<bool>("info.hidden", false));
				info_group_box->setHidden(ptree.get<bool>("info.hidden", false));
				theme_index_ = ptree.get<int>("info.theme_type", theme_index_);

				// Autofocus
				compute_desc_.autofocus_size = ptree.get<int>("autofocus.size", compute_desc_.autofocus_size);
				compute_desc_.autofocus_z_min = ptree.get<float>("autofocus.z_min", compute_desc_.autofocus_z_min);
				compute_desc_.autofocus_z_max = ptree.get<float>("autofocus.z_max", compute_desc_.autofocus_z_max);
				compute_desc_.autofocus_z_div = ptree.get<uint>("autofocus.steps", compute_desc_.autofocus_z_div);
				compute_desc_.autofocus_z_iter = ptree.get<uint>("autofocus.loops", compute_desc_.autofocus_z_iter);

				//flowgraphy
				uint flowgraphy_level = ptree.get<uint>("flowgraphy.level", compute_desc_.flowgraphy_level);
				compute_desc_.flowgraphy_level = (flowgraphy_level % 2 == 0) ? (flowgraphy_level + 1) : (flowgraphy_level);
				compute_desc_.flowgraphy_enabled = ptree.get<bool>("flowgraphy.enable", compute_desc_.flowgraphy_enabled);

				// Reset button
				config.set_cuda_device = ptree.get<bool>("reset.set_cuda_device", config.set_cuda_device);
				config.auto_device_number = ptree.get<bool>("reset.auto_device_number", config.auto_device_number);
				config.device_number = ptree.get<int>("reset.device_number", config.device_number);

				// Composite
				compute_desc_.component_r.p_min = ptree.get<ushort>("composite.pmin", 0);
				compute_desc_.component_b.p_max = ptree.get<ushort>("composite.pmax", 0);
				compute_desc_.component_r.weight = ptree.get<float>("composite.weight_r", 1);
				compute_desc_.component_g.weight = ptree.get<float>("composite.weight_g", 1);
				compute_desc_.component_b.weight = ptree.get<float>("composite.weight_b", 1);
				compute_desc_.composite_auto_weights_ = ptree.get<bool>("composite.auto_weights", false);

				// Interpolation
				compute_desc_.interpolation_enabled = ptree.get<bool>("interpolation.enabled", false);
				compute_desc_.interp_lambda1 = ptree.get<float>("interpolation.lambda1", 870) * 1.0e-9f;
				compute_desc_.interp_lambda2 = ptree.get<float>("interpolation.lambda2", 820) * 1.0e-9f;
				compute_desc_.interp_sensitivity = ptree.get<float>("interpolation.sensitivity", 0.9f);
				compute_desc_.interp_shift = ptree.get<int>("interpolation.shift", 0);

				notify();
			}
		}

		void MainWindow::save_ini(const std::string& path)
		{
			boost::property_tree::ptree ptree;
			GroupBox *image_rendering_group_box = ui.ImageRenderingGroupBox;
			GroupBox *view_group_box = ui.ViewGroupBox;
			GroupBox *special_group_box = ui.PostProcessingGroupBox;
			GroupBox *record_group_box = ui.RecordGroupBox;
			GroupBox *import_group_box = ui.ImportGroupBox;
			GroupBox *info_group_box = ui.InfoGroupBox;
			Config& config = global::global_config;
			
			// Config
			ptree.put<uint>("config.input_buffer_size", config.input_queue_max_size);
			ptree.put<uint>("config.output_buffer_size", config.output_queue_max_size);
			ptree.put<uint>("config.float_buffer_size", config.float_queue_max_size);
			ptree.put<uint>("config.input_file_buffer_size", config.reader_buf_max_size);
			ptree.put<uint>("config.stft_cuts_output_buffer_size", config.stft_cuts_output_buffer_size);
			ptree.put<int>("config.stft_buffer_size", compute_desc_.stft_level);
			ptree.put<int>("config.reference_buffer_size", compute_desc_.ref_diff_level);
			ptree.put<uint>("config.accumulation_buffer_size", compute_desc_.img_acc_slice_xy_level);
			ptree.put<int>("config.convolution_buffer_size", compute_desc_.special_buffer_size);
			ptree.put<uint>("config.frame_timeout", config.frame_timeout);
			ptree.put<bool>("config.flush_on_refresh", config.flush_on_refresh);
			ptree.put<ushort>("config.display_rate", static_cast<ushort>(compute_desc_.display_rate));

			// Image rendering
			ptree.put<bool>("image_rendering.hidden", image_rendering_group_box->isHidden());
			ptree.put("image_rendering.camera", kCamera);
			ptree.put<ushort>("image_rendering.phase_number", compute_desc_.nsamples);
			ptree.put<ushort>("image_rendering.p_index", compute_desc_.pindex);
			ptree.put<float>("image_rendering.lambda", compute_desc_.lambda);
			ptree.put<float>("image_rendering.z_distance", compute_desc_.zdistance);
			ptree.put<double>("image_rendering.z_step", z_step_);
			ptree.put<holovibes::Algorithm>("image_rendering.algorithm", compute_desc_.algorithm);
			
			// View
			ptree.put<bool>("view.hidden", view_group_box->isHidden());
			ptree.put<holovibes::ImgType>("view.view_mode", compute_desc_.img_type);
			ptree.put<bool>("view.log_scale_enabled", compute_desc_.log_scale_slice_xy_enabled);
			ptree.put<bool>("view.log_scale_enabled_cut_xz", compute_desc_.log_scale_slice_xz_enabled);
			ptree.put<bool>("view.log_scale_enabled_cut_yz", compute_desc_.log_scale_slice_yz_enabled);
			ptree.put<bool>("view.shift_corners_enabled", compute_desc_.shift_corners_enabled);
			ptree.put<bool>("view.contrast_enabled", compute_desc_.contrast_enabled);
			ptree.put<float>("view.contrast_min", compute_desc_.contrast_min_slice_xy);
			ptree.put<float>("view.contrast_max", compute_desc_.contrast_max_slice_xy);
			ptree.put<ushort>("view.cuts_contrast_p_offset", compute_desc_.cuts_contrast_p_offset);
			ptree.put<bool>("view.accumulation_enabled", compute_desc_.img_acc_slice_xy_enabled);
			ptree.put<float>("view.mainWindow_rotate", displayAngle);
			ptree.put<float>("view.xCut_rotate", xzAngle);
			ptree.put<float>("view.yCut_rotate", yzAngle);
			ptree.put<int>("view.mainWindow_flip", displayFlip);
			ptree.put<int>("view.xCut_flip", xzFlip);
			ptree.put<int>("view.yCut_flip", yzFlip);

			// Post-processing
			ptree.put<bool>("post_processing.hidden", special_group_box->isHidden());
			ptree.put<ushort>("post_processing.image_ratio_q", compute_desc_.vibrometry_q);
			ptree.put<bool>("post_processing.average_enabled", is_enabled_average_);

			// Record
			ptree.put<bool>("record.hidden", record_group_box->isHidden());

			// Import
			ptree.put<bool>("import.hidden", import_group_box->isHidden());
			ptree.put<float>("import.pixel_size", compute_desc_.pixel_size);

			// Info
			ptree.put<bool>("info.hidden", info_group_box->isHidden());
			ptree.put<ushort>("info.theme_type", theme_index_);

			// Autofocus
			ptree.put<uint>("autofocus.size", compute_desc_.autofocus_size);
			ptree.put<float>("autofocus.z_min", compute_desc_.autofocus_z_min);
			ptree.put<float>("autofocus.z_max", compute_desc_.autofocus_z_max);
			ptree.put<uint>("autofocus.steps", compute_desc_.autofocus_z_div);
			ptree.put<uint>("autofocus.loops", compute_desc_.autofocus_z_iter);

			// Composite
			ptree.put<ushort>("composite.pmin", compute_desc_.component_r.p_min);
			ptree.put<ushort>("composite.pmax", compute_desc_.component_b.p_max);
			ptree.put<float>("composite.weight_r", compute_desc_.component_r.weight);
			ptree.put<float>("composite.weight_g", compute_desc_.component_g.weight);
			ptree.put<float>("composite.weight_b", compute_desc_.component_b.weight);
			ptree.put<bool>("composite.auto_weights", compute_desc_.composite_auto_weights_);

			//flowgraphy
			ptree.put<uint>("flowgraphy.level", compute_desc_.flowgraphy_level);
			ptree.put<bool>("flowgraphy.enable", compute_desc_.flowgraphy_enabled);

			//Reset
			ptree.put<bool>("reset.set_cuda_device", config.set_cuda_device);
			ptree.put<bool>("reset.auto_device_number", config.auto_device_number);
			ptree.put<uint>("reset.device_number", config.device_number);

			// Interpolation
			ptree.put<bool>("interpolation.enabled", compute_desc_.interpolation_enabled);
			ptree.put<float>("interpolation.lambda1", compute_desc_.interp_lambda1 * 1.0e9);
			ptree.put<float>("interpolation.lambda2", compute_desc_.interp_lambda2 * 1.0e9);
			ptree.put<float>("interpolation.sensitivity", compute_desc_.interp_sensitivity);
			ptree.put<int>("interpolation.shift", compute_desc_.interp_shift);

			
			boost::property_tree::write_ini(holovibes_.get_launch_path() + "/" + path, ptree);
		}

		void MainWindow::open_file(const std::string& path)
		{
			QDesktopServices::openUrl(QUrl::fromLocalFile(QString(path.c_str())));
		}
		#pragma endregion
		/* ------------ */
		#pragma region Close Compute
		void MainWindow::close_critical_compute()
		{ 
			if (compute_desc_.average_enabled)
				set_average_mode(false);
			cancel_stft_view(compute_desc_);
			if (compute_desc_.ref_diff_enabled || compute_desc_.ref_sliding_enabled)
				cancel_take_reference();
			if (compute_desc_.filter_2d_enabled)
				cancel_filter2D();
			holovibes_.dispose_compute();
		}

		void MainWindow::camera_none()
		{
			close_critical_compute();
			if (!is_direct_mode())
				holovibes_.dispose_compute();
			holovibes_.dispose_capture();
			close_windows();
			remove_infos();
			ui.actionSettings->setEnabled(false);
			is_enabled_camera_ = false;
			compute_desc_.compute_mode = Computation::Stop;
			notify();
		}

		void MainWindow::remove_infos()
		{
			try
			{
				InfoManager *manager = InfoManager::get_manager();
				manager->clear_infos();
			}
			catch (std::exception& e)
			{
				std::cerr << e.what() << std::endl;
			}
		}

		void MainWindow::close_windows()
		{
			sliceXZ.reset(nullptr);
			sliceYZ.reset(nullptr);
			plot_window_.reset(nullptr);
			mainDisplay.reset(nullptr);
			lens_window.reset(nullptr);
		}

		void MainWindow::reset()
		{
			Config&	config = global::global_config;
			int		device = 0;

			close_critical_compute();
			camera_none();
			InfoManager *manager = InfoManager::get_manager();
			manager->update_info("Status", "Resetting...");
			qApp->processEvents();
			if (!is_direct_mode())
				holovibes_.dispose_compute();
			holovibes_.dispose_capture();
			compute_desc_.pindex = 0;
			compute_desc_.nsamples = 1;
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
			remove_infos();
			try
			{
				load_ini(GLOBAL_INI_PATH);
			}
			catch (std::exception&)
			{
				std::cout << GLOBAL_INI_PATH << ": Config file not found. It will use the default values." << std::endl;
			}
			notify();
		}

		void MainWindow::closeEvent(QCloseEvent* event)
		{
			if (compute_desc_.compute_mode != Computation::Stop)
				close_critical_compute();
			camera_none();
			close_windows();
			remove_infos();
			// Avoiding "unused variable" warning.
			static_cast<void*>(event);
			save_ini(GLOBAL_INI_PATH);
		}
		#pragma endregion
		/* ------------ */
		#pragma region Cameras
		void MainWindow::change_camera(CameraKind c)
		{
			close_critical_compute();
			close_windows();
			remove_infos();
			if (c != CameraKind::NONE)
			{
				try
				{
					mainDisplay.reset(nullptr);
					if (!is_direct_mode())
						holovibes_.dispose_compute();
					holovibes_.dispose_capture();
					holovibes_.init_capture(c);
					is_enabled_camera_ = true;
					set_image_mode();
					import_type_ = ImportType::Camera;
					kCamera = c;
					QAction* settings = ui.actionSettings;
					settings->setEnabled(true);
					notify();
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

		void MainWindow::camera_ids()
		{
			change_camera(CameraKind::IDS);
		}

		void MainWindow::camera_ixon()
		{
			change_camera(CameraKind::Ixon);
		}

		void MainWindow::camera_hamamatsu()
		{
			change_camera(CameraKind::Hamamatsu);
		}

		void MainWindow::camera_adimec()
		{
			change_camera(CameraKind::Adimec);
		}

		void MainWindow::camera_edge()
		{
			change_camera(CameraKind::Edge);
		}

		void MainWindow::camera_pike()
		{
			change_camera(CameraKind::Pike);
		}

		void MainWindow::camera_pixelfly()
		{
			change_camera(CameraKind::Pixelfly);
		}

		void MainWindow::camera_xiq()
		{
			change_camera(CameraKind::xiQ);
		}

		void MainWindow::camera_xib()
		{
			change_camera(CameraKind::xiB);
		}

		void MainWindow::camera_photon_focus()
		{
			change_camera(CameraKind::PhotonFocus);
		}

		void MainWindow::configure_camera()
		{
			open_file(boost::filesystem::current_path().generic_string() + "/" + holovibes_.get_camera_ini_path());
		}
		#pragma endregion
		/* ------------ */
		#pragma region Image Mode
		void MainWindow::init_image_mode(QPoint& position, QSize& size)
		{
			if (mainDisplay)
			{
				position = mainDisplay->framePosition();
				size = mainDisplay->size();
				mainDisplay.reset(nullptr);
			}
		}

		void MainWindow::set_direct_mode()
		{
			close_critical_compute();
			close_windows();
			InfoManager::get_manager()->remove_info("Throughput");
			compute_desc_.compute_mode = Computation::Stop;
			notify();
			if (is_enabled_camera_)
			{
				QPoint pos(0, 0);
				QSize size(512, 512);
				init_image_mode(pos, size);
				compute_desc_.compute_mode = Computation::Direct;
				createPipe();
				mainDisplay.reset(
					new DirectWindow(
						pos, size,
						holovibes_.get_capture_queue()));
				mainDisplay->setTitle(QString("XY view"));
				mainDisplay->setCd(&compute_desc_);
				const FrameDescriptor& fd = holovibes_.get_capture_queue().get_frame_desc();
				InfoManager::get_manager()->insertInputSource(fd.width, fd.height, fd.depth);
				set_convolution_mode(false);
				notify();
			}
		}

		void MainWindow::createPipe()
		{

			uint depth = holovibes_.get_capture_queue().get_frame_desc().depth;
			
			if (compute_desc_.compute_mode == Computation::Hologram)
			{
				depth = 2;
				if (compute_desc_.img_type == ImgType::Complex)
					depth = 8;
				else if (compute_desc_.img_type == ImgType::Composite)
					depth = 6;
			}
			/* ---------- */
			try
			{
				holovibes_.init_compute(ThreadCompute::PipeType::PIPE, depth);
				while (!holovibes_.get_pipe());
				holovibes_.get_pipe()->register_observer(*this);
			}
			catch (std::runtime_error& e)
			{
				std::cerr << "cannot create Pipe :" << std::endl;
				std::cerr << e.what() << std::endl;
			}
		}

		void MainWindow::createHoloWindow()
		{
			QPoint pos(0, 0);
			QSize size(512, 512);
			init_image_mode(pos, size);
			/* ---------- */
			try
			{
				mainDisplay.reset(
					new HoloWindow(
						pos, size,
						holovibes_.get_output_queue(),
						holovibes_.get_pipe(),
						sliceXZ,
						sliceYZ,
						this));
				mainDisplay->setTitle(QString("XY view"));
				mainDisplay->setCd(&compute_desc_);
				mainDisplay->resetTransform();
				mainDisplay->setAngle(displayAngle);
				mainDisplay->setFlip(displayFlip);
			}
			catch (std::runtime_error& e)
			{
				std::cerr << "error createHoloWindow :" << std::endl;
				std::cerr << e.what() << std::endl;
			}
		}

		void MainWindow::set_holographic_mode()
		{
			close_critical_compute();
			close_windows();
			/* ---------- */
			try
			{
				compute_desc_.compute_mode = Computation::Hologram;
				/* ---------- */
				createPipe();
				createHoloWindow();
				/* ---------- */
				const FrameDescriptor& fd = holovibes_.get_output_queue().get_frame_desc();
				InfoManager::get_manager()->insertInputSource(fd.width, fd.height, fd.depth);
				/* ---------- */
				compute_desc_.contrast_enabled = true;
				set_auto_contrast();
				notify();
			}
			catch (std::runtime_error& e)
			{
				std::cerr << "cannot set holographic mode :" << std::endl;
				std::cerr << e.what() << std::endl;
			}
		}

		void MainWindow::refreshViewMode()
		{
			close_critical_compute();
			close_windows();
			try
			{
				createPipe();
				createHoloWindow();
			}
			catch (std::runtime_error& e)
			{
				mainDisplay.reset(nullptr);
				std::cerr << "error refreshViewMode :" << std::endl;
				std::cerr << e.what() << std::endl;
			}
			notify();
		}

		namespace
		{
			bool need_refresh(const QString& last_type, const QString& new_type)
			{
				std::vector<QString> types_needing_refresh({ "Complex output", "Composite image" });
				for (auto& type : types_needing_refresh)
					if ((last_type == type) != (new_type == type))
						return true;
				return false;
			}
		}
		void MainWindow::set_view_mode(const QString value)
		{
			if (!is_direct_mode())
			{
				QComboBox* ptr = ui.ViewModeComboBox;

				compute_desc_.img_type = static_cast<ImgType>(ptr->currentIndex());
				if (need_refresh(last_img_type_, value))
				{
					refreshViewMode();
					if (compute_desc_.stft_view_enabled)
						set_auto_contrast_cuts();
				}
				last_img_type_ = value;
				layout_toggled();

				set_auto_contrast();
				notify();
			}
		}
		
		bool MainWindow::is_direct_mode()
		{
			return compute_desc_.compute_mode == Computation::Direct;
		}

		void MainWindow::set_image_mode()
		{
			if (compute_desc_.compute_mode == Computation::Direct)
				set_direct_mode();
			else if (compute_desc_.compute_mode == Computation::Hologram)
				set_holographic_mode();
			else
			{
				if (ui.DirectRadioButton->isChecked())
					set_direct_mode();
				else
					set_holographic_mode();
			}
		}
		#pragma endregion
		/* ------------ */
		#pragma region STFT
		void MainWindow::cancel_stft_slice_view()
		{
			InfoManager *manager = InfoManager::get_manager();

			manager->remove_info("STFT Slice Cursor");

			compute_desc_.contrast_max_slice_xz = false;
			compute_desc_.contrast_max_slice_yz = false;
			compute_desc_.log_scale_slice_xz_enabled = false;
			compute_desc_.log_scale_slice_yz_enabled = false;
			compute_desc_.img_acc_slice_xz_enabled = false;
			compute_desc_.img_acc_slice_yz_enabled = false;
			holovibes_.get_pipe()->delete_stft_slice_queue();
			while (holovibes_.get_pipe()->get_cuts_delete_request());
			compute_desc_.stft_view_enabled = false;
			sliceXZ.reset(nullptr);
			sliceYZ.reset(nullptr);

			ui.STFTCutsCheckBox->setChecked(false);

			mainDisplay->setCursor(Qt::ArrowCursor);
			mainDisplay->resetSelection();

			notify();
		}

		void MainWindow::set_crop_stft(bool b)
		{
			if (!is_direct_mode())
			{
				compute_desc_.croped_stft = b;
				std::stringstream ss;
				ss << "(X1,X2,X3,X4) = (";
				if (b)
				{
					auto zone = compute_desc_.getZoomedZone();
					ss << zone.x() << "," << zone.y() << "," << zone.right() << "," << zone.bottom() << ")";
				}
				else
				{
					auto fd = holovibes_.get_cam_frame_desc();
					ss << "0,0," << fd.width - 1 << "," << fd.height - 1 << ")";
				}
				InfoManager::get_manager()->update_info("STFT Zone", ss.str());
				holovibes_.get_pipe()->request_update_n(compute_desc_.nsamples);
			}
		}

		void MainWindow::update_stft_steps(int value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.stft_steps = value;
				notify();
			}
		}

		void MainWindow::stft_view(bool checked)
		{
			InfoManager *manager = InfoManager::get_manager();
			manager->insert_info(InfoManager::InfoType::STFT_SLICE_CURSOR, "STFT Slice Cursor", "(Y,X) = (0,0)");

			QComboBox* winSelection = ui.WindowSelectionComboBox;
			winSelection->setEnabled(checked);
			winSelection->setCurrentIndex((!checked) ? 0 : winSelection->currentIndex());
			if (checked)
			{
				try
				{
					if (compute_desc_.filter_2d_enabled)
						cancel_filter2D();
					holovibes_.get_pipe()->create_stft_slice_queue();
					// set positions of new windows according to the position of the main GL window
					QPoint			xzPos = mainDisplay->framePosition() + QPoint(0, mainDisplay->height() + 42);
					QPoint			yzPos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 20, 0);
					const ushort	nImg = compute_desc_.nsamples;
					const uint		nSize = (nImg < 128 ? 128 : (nImg > 256 ? 256 : nImg)) * 2;

					while (holovibes_.get_pipe()->get_update_n_request());
 					while (holovibes_.get_pipe()->get_cuts_request());
					sliceXZ.reset(nullptr);
					sliceXZ.reset(new SliceWindow(
						xzPos,
						QSize(mainDisplay->width(), nSize),
						holovibes_.get_pipe()->get_stft_slice_queue(0),
						KindOfView::SliceXZ,
						this));
					sliceXZ->setTitle("XZ view");
					sliceXZ->setAngle(xzAngle);
					sliceXZ->setFlip(xzFlip);
					sliceXZ->setCd(&compute_desc_);
					
					sliceYZ.reset(nullptr);
					sliceYZ.reset(new SliceWindow(
						yzPos,
						QSize(nSize, mainDisplay->height()),
						holovibes_.get_pipe()->get_stft_slice_queue(1),
						KindOfView::SliceYZ,
						this));
					sliceYZ->setTitle("YZ view");
					sliceYZ->setAngle(yzAngle);
					sliceYZ->setFlip(yzFlip);
					sliceYZ->setCd(&compute_desc_);

					mainDisplay->getOverlayManager().create_overlay<Cross>();
					compute_desc_.stft_view_enabled = true;
					compute_desc_.average_enabled = false;
					
					set_auto_contrast_cuts();
					auto holo = dynamic_cast<HoloWindow*>(mainDisplay.get());
					if (holo)
						holo->update_slice_transforms();
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

		void MainWindow::cancel_stft_view(ComputeDescriptor& cd)
		{
			if (compute_desc_.stft_view_enabled)
				cancel_stft_slice_view();
			if (compute_desc_.p_accu_enabled)
				compute_desc_.p_accu_enabled = false;
			compute_desc_.stft_view_enabled = false;
			notify();
		}

		#pragma endregion
		/* ------------ */
		#pragma region Computation
		void MainWindow::change_window()
		{
			QComboBox *window_cbox = ui.WindowSelectionComboBox;

			if (window_cbox->currentIndex() == 0)
				compute_desc_.current_window = WindowKind::XYview;
			else if (window_cbox->currentIndex() == 1)
				compute_desc_.current_window = WindowKind::XZview;
			else if (window_cbox->currentIndex() == 2)
				compute_desc_.current_window = WindowKind::YZview;
			notify();
		}

		void MainWindow::set_convolution_mode(const bool value)
		{
			if (value == true && compute_desc_.convo_matrix.empty())
			{
				display_error("No valid kernel has been given");
				compute_desc_.convolution_enabled = false;
			}
			else
			{
				compute_desc_.convolution_enabled = value;
				set_auto_contrast();
			}
			notify();
		}

		void MainWindow::set_flowgraphy_mode(const bool value)
		{
			compute_desc_.flowgraphy_enabled = value;
			if (!is_direct_mode())
				pipe_refresh();
			notify();
		}

		void MainWindow::take_reference()
		{
			if (!is_direct_mode())
			{
				compute_desc_.ref_diff_enabled = true;
				holovibes_.get_pipe()->request_ref_diff_refresh();
				InfoManager::get_manager()->update_info("Reference", "Processing... ");
				notify();
			}
		}

		void MainWindow::take_sliding_ref()
		{
			if (!is_direct_mode())
			{
				compute_desc_.ref_sliding_enabled = true;
				holovibes_.get_pipe()->request_ref_diff_refresh();
				InfoManager::get_manager()->update_info("Reference", "Processing...");
				notify();
			}
		}

		void MainWindow::cancel_take_reference()
		{
			if (!is_direct_mode())
			{
				compute_desc_.ref_diff_enabled = false;
				compute_desc_.ref_sliding_enabled = false;
				holovibes_.get_pipe()->request_ref_diff_refresh();
				InfoManager::get_manager()->remove_info("Reference");
				notify();
			}
		}

		void MainWindow::set_filter2D()
		{
			if (!is_direct_mode())
			{
				mainDisplay->resetTransform();
				mainDisplay->getOverlayManager().create_overlay<Filter2D>();
				ui.Filter2DPushButton->setStyleSheet("QPushButton {color: #009FFF;}");
				compute_desc_.log_scale_slice_xy_enabled = true;
				compute_desc_.shift_corners_enabled = false;
				compute_desc_.filter_2d_enabled = true;
				set_auto_contrast();
				InfoManager::get_manager()->update_info("Filter2D", "Processing...");
				notify();
			}
		}

		void MainWindow::cancel_filter2D()
		{
			if (!is_direct_mode())
			{
				InfoManager::get_manager()->remove_info("Filter2D");
				compute_desc_.filter_2d_enabled = false;
				compute_desc_.log_scale_slice_xy_enabled = false;
				compute_desc_.setStftZone(units::RectFd());
				mainDisplay->getOverlayManager().disable_all(Filter2D);
				mainDisplay->getOverlayManager().create_default();
				mainDisplay->resetTransform();
				set_auto_contrast();
				notify();
			}
		}

		void MainWindow::set_shifted_corners(const bool value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.shift_corners_enabled = value;
				pipe_refresh();
			}
		}

		void MainWindow::setPhase()
		{
			if (!is_direct_mode())
			{
				int phaseNumber = ui.PhaseNumberSpinBox->value();
				phaseNumber = std::max(1, phaseNumber);

				if (phaseNumber == compute_desc_.nsamples)
					return;
				compute_desc_.nsamples = phaseNumber;
				notify();
				holovibes_.get_pipe()->request_update_n(phaseNumber);
				while (holovibes_.get_pipe()->get_request_refresh());
				set_p_accu();
			}
		}

		void MainWindow::set_special_buffer_size(int value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.special_buffer_size = value;
				if (compute_desc_.special_buffer_size < static_cast<std::atomic<int>>(compute_desc_.flowgraphy_level))
				{
					if (compute_desc_.special_buffer_size % 2 == 0)
						compute_desc_.flowgraphy_level = compute_desc_.special_buffer_size - 1;
					else
						compute_desc_.flowgraphy_level = compute_desc_.special_buffer_size;
				}
				notify();
				set_auto_contrast();
			}
		}

		void MainWindow::update_lens_view(bool value)
		{
			if (value)
			{
				try
				{
					// set positions of new windows according to the position of the main GL window
					QPoint			pos = mainDisplay->framePosition() + QPoint(mainDisplay->height() + 300, 0);
					auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
					if (pipe)
					{
						lens_window.reset(new DirectWindow(
							pos,
							QSize(mainDisplay->width(), mainDisplay->height()),
							*pipe->get_lens_queue()));
					}
					lens_window->setTitle("Lens view");
					lens_window->setCd(&compute_desc_);
				}
				catch (std::exception& e)
				{
					std::cerr << e.what() << std::endl;
					cancel_stft_slice_view();
				}
			}
			else
			{
				lens_window = nullptr;
			}
			compute_desc_.gpu_lens_display_enabled = value;
			pipe_refresh();
		}

		void MainWindow::set_p_accu()
		{
			auto spinbox = ui.PAccSpinBox;
			auto checkBox = ui.PAccuCheckBox;
			if (compute_desc_.p_accu_enabled != checkBox->isChecked())
				pipe_refresh();
			compute_desc_.p_accu_enabled = checkBox->isChecked();
			compute_desc_.p_acc_level = spinbox->value();

			notify();
		}

		void MainWindow::set_x_accu()
		{
			auto box = ui.XAccSpinBox;
			auto checkBox = ui.XAccuCheckBox;
			compute_desc_.x_accu_enabled = checkBox->isChecked();
			compute_desc_.x_acc_level = box->value();
			notify();
		}

		void MainWindow::set_y_accu()
		{
			auto box = ui.YAccSpinBox;
			auto checkBox = ui.YAccuCheckBox;
			compute_desc_.y_accu_enabled = checkBox->isChecked();
			compute_desc_.y_acc_level = box->value();
			notify();
		}

		void MainWindow::set_p(int value)
		{
			if (!is_direct_mode())
			{
				if (value < static_cast<int>(compute_desc_.nsamples))
				{
					compute_desc_.pindex = value;
					notify();
				}
				else
					display_error("p param has to be between 1 and #img");
			}
		}
		void MainWindow::set_composite_intervals()
		{
			QSpinBox *min_box = ui.PMinSpinBox_Composite;
			QSpinBox *max_box = ui.PMaxSpinBox_Composite;

			unsigned short pmin = min_box->value();
			unsigned short pmax = max_box->value();

			compute_desc_.component_r.p_min = pmin;
			compute_desc_.component_b.p_max = pmax;
			notify();
		}

		void MainWindow::set_composite_weights()
		{
			QDoubleSpinBox *boxes[3];
			boxes[0] = ui.WeightSpinBox_R;
			boxes[1] = ui.WeightSpinBox_G;
			boxes[2] = ui.WeightSpinBox_B;
			Component *components[] = { &compute_desc_.component_r, &compute_desc_.component_g, &compute_desc_.component_b };
			for (int i = 0; i < 3; i++)
				components[i]->weight = boxes[i]->value();
		}

		void MainWindow::set_composite_auto_weights(bool value)
		{
			compute_desc_.composite_auto_weights_ = value;
			set_auto_contrast();
		}


		void MainWindow::set_flowgraphy_level(const int value)
		{
			int flag = 0;

			if (!is_direct_mode())
			{
				if (value % 2 == 0)
				{
					if (value + 1 <= compute_desc_.special_buffer_size)
					{
						compute_desc_.flowgraphy_level = value + 1;
						flag = 1;
					}
				}
				else
				{
					if (value <= compute_desc_.special_buffer_size)
					{
						compute_desc_.flowgraphy_level = value;
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

				if (compute_desc_.pindex < compute_desc_.nsamples)
				{
					compute_desc_.pindex = compute_desc_.pindex + 1;
					notify();
					set_auto_contrast();
				}
				else
					display_error("p param has to be between 1 and #img");
			}
		}

		void MainWindow::decrement_p()
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.pindex > 0)
				{
					compute_desc_.pindex = compute_desc_.pindex - 1;
					notify();
					set_auto_contrast();
				}
				else
					display_error("p param has to be between 1 and #img");
			}
		}

		void MainWindow::set_wavelength(const double value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.lambda = static_cast<float>(value) * 1.0e-9f;
				pipe_refresh();
			}
		}

		void MainWindow::set_interp_lambda1(const double value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.interp_lambda1 = static_cast<float>(value) * 1.0e-9f;
				pipe_refresh();
			}
		}

		void MainWindow::set_interp_lambda2(const double value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.interp_lambda2 = static_cast<float>(value) * 1.0e-9f;
				pipe_refresh();
			}
		}

		void MainWindow::set_interp_sensitivity(const double value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.interp_sensitivity = value;
				pipe_refresh();
			}
		}

		void MainWindow::set_interp_shift(const int value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.interp_shift = value;
				pipe_refresh();
			}
		}

		void MainWindow::set_interpolation(bool value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.interpolation_enabled = value;
				pipe_refresh();
			}
		}

		void MainWindow::set_z(const double value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.zdistance = static_cast<float>(value);
				pipe_refresh();
			}
		}

		void MainWindow::increment_z()
		{
			if (!is_direct_mode())
			{
				set_z(compute_desc_.zdistance + z_step_);
				ui.ZDoubleSpinBox->setValue(compute_desc_.zdistance);
			}
		}

		void MainWindow::decrement_z()
		{
			if (!is_direct_mode())
			{
				set_z(compute_desc_.zdistance - z_step_);
				ui.ZDoubleSpinBox->setValue(compute_desc_.zdistance);
			}
		}

		void MainWindow::set_z_step(const double value)
		{
			z_step_ = value;
			ui.ZDoubleSpinBox->setSingleStep(value);
		}

		void MainWindow::set_algorithm(const QString value)
		{
			if (!is_direct_mode())
			{
				if (value == "None")
					compute_desc_.algorithm = Algorithm::None;
				else if (value == "1FFT")
					compute_desc_.algorithm = Algorithm::FFT1;
				else if (value == "2FFT")
					compute_desc_.algorithm = Algorithm::FFT2;
				else
					assert(!"Unknow Algorithm.");
				notify();
				set_auto_contrast();
			}
		}

		void MainWindow::set_unwrap_history_size(int value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.unwrap_history_size = value;
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
				if (compute_desc_.current_window == WindowKind::XYview)
					compute_desc_.img_acc_slice_xy_enabled = value;
				else if (compute_desc_.current_window == WindowKind::XZview)
					compute_desc_.img_acc_slice_xz_enabled = value;
				else if (compute_desc_.current_window == WindowKind::YZview)
					compute_desc_.img_acc_slice_yz_enabled = value;
				holovibes_.get_pipe()->request_acc_refresh();
				notify();
			}
		}

		void MainWindow::set_accumulation_level(int value)
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.current_window == WindowKind::XYview)
					compute_desc_.img_acc_slice_xy_level = value;
				else if (compute_desc_.current_window == WindowKind::XZview)
					compute_desc_.img_acc_slice_xz_level = value;
				else if (compute_desc_.current_window == WindowKind::YZview)
					compute_desc_.img_acc_slice_yz_level = value;
				holovibes_.get_pipe()->request_acc_refresh();
			}
		}

		void MainWindow::set_xy_stabilization_enable(bool value)
		{
			compute_desc_.xy_stabilization_enabled = value;
			pipe_refresh();
		}

		void MainWindow::set_import_pixel_size(const double value)
		{
			compute_desc_.pixel_size = value;
		}

		void MainWindow::set_z_iter(const int value)
		{
			if (!is_direct_mode())
				compute_desc_.autofocus_z_iter = value;
			notify();
		}

		void MainWindow::set_z_div(const int value)
		{
			if (!is_direct_mode())
				compute_desc_.autofocus_z_div = value;
			notify();
		}

		void MainWindow::pipe_refresh()
		{
			if (!is_direct_mode())
			{
				try
				{
					if (!holovibes_.get_pipe()->get_request_refresh())
						holovibes_.get_pipe()->request_refresh();
				}
				catch (std::runtime_error& e)
				{
					std::cerr << e.what() << std::endl;
				}
			}
		}

		void MainWindow::set_stabilization_area()
		{
			mainDisplay->getOverlayManager().create_overlay<Stabilization>();
		}

		void MainWindow::set_composite_area()
		{
			mainDisplay->getOverlayManager().create_overlay<CompositeArea>();
		}

		#pragma endregion
		/* ------------ */
		#pragma region Texture
		void MainWindow::rotateTexture()
		{
			WindowKind curWin = compute_desc_.current_window;

			if (curWin == WindowKind::XYview)
			{
				displayAngle = (displayAngle == 270.f) ? 0.f : displayAngle + 90.f;
				mainDisplay->setAngle(displayAngle);
			}
			else if (sliceXZ && curWin == WindowKind::XZview)
			{
				xzAngle = (xzAngle == 270.f) ? 0.f : xzAngle + 90.f;
				sliceXZ->setAngle(xzAngle);
			}
			else if (sliceYZ && curWin == WindowKind::YZview)
			{
				yzAngle = (yzAngle == 270.f) ? 0.f : yzAngle + 90.f;
				sliceYZ->setAngle(yzAngle);
			}
			notify();
		}

		void MainWindow::flipTexture()
		{
			WindowKind curWin = compute_desc_.current_window;

			if (curWin == WindowKind::XYview)
			{
				displayFlip = !displayFlip;
				mainDisplay->setFlip(displayFlip);
			}
			else if (sliceXZ && curWin == WindowKind::XZview)
			{
				xzFlip = !xzFlip;
				sliceXZ->setFlip(xzFlip);
			}
			else if (sliceYZ && curWin == WindowKind::YZview)
			{
				yzFlip = !yzFlip;
				sliceYZ->setFlip(yzFlip);
			}
			notify();
		}

		void MainWindow::set_scale_bar(bool value)
		{
			if (value)
			{
				mainDisplay->getOverlayManager().create_overlay<Scale>();
			}
			else
			{
				mainDisplay->getOverlayManager().disable_all(Scale);
			}

		}

		void MainWindow::set_scale_bar_correction_factor(double value)
		{
			compute_desc_.scale_bar_correction_factor = value;
		}
		#pragma endregion
		/* ------------ */
		#pragma region Autofocus
		void MainWindow::set_autofocus_mode()
		{
			// If current overlay is Autofocus, disable it
			if (mainDisplay->getKindOfOverlay() == Autofocus)
			{
				mainDisplay->getOverlayManager().disable_all(Autofocus);
				mainDisplay->getOverlayManager().create_default();
				notify();
			}
			else if (compute_desc_.autofocus_z_min >= compute_desc_.autofocus_z_max)
				display_error("z min have to be strictly inferior to z max");
			else
			{
				mainDisplay->getOverlayManager().create_overlay<Autofocus>();
				notify();
			}
		}

		void MainWindow::set_xy_stabilization_show_convolution(bool value)
		{
			compute_desc_.xy_stabilization_show_convolution = value;
			pipe_refresh();
			while (holovibes_.get_pipe()->get_refresh_request())
				continue;
			set_auto_contrast();
		}

		void MainWindow::set_z_min(const double value)
		{
			if (!is_direct_mode())
				compute_desc_.autofocus_z_min = value;
		}

		void MainWindow::set_z_max(const double value)
		{
			if (!is_direct_mode())
				compute_desc_.autofocus_z_max = value;
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
		#pragma endregion
		/* ------------ */
		#pragma region Contrast - Log
		void MainWindow::set_contrast_mode(bool value)
		{
			if (!is_direct_mode())
			{
				change_window();
				compute_desc_.contrast_enabled = value;
				set_contrast_min(ui.ContrastMinDoubleSpinBox->value());
				set_contrast_max(ui.ContrastMaxDoubleSpinBox->value());
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_auto_contrast_cuts()
		 {
			WindowKind current_window = compute_desc_.current_window;
			compute_desc_.current_window = WindowKind::XZview;
			set_auto_contrast();
			while (holovibes_.get_pipe()->get_autocontrast_request());
			compute_desc_.current_window = WindowKind::YZview;
			set_auto_contrast();
			compute_desc_.current_window = current_window;
		}

		void MainWindow::set_auto_contrast()
		{
			if (!is_direct_mode() &&
				!compute_desc_.flowgraphy_enabled)
			{
				try
				{
					holovibes_.get_pipe()->request_autocontrast();
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
				if (compute_desc_.contrast_enabled)
				{
					if (compute_desc_.current_window == WindowKind::XYview)
					{
						if (compute_desc_.log_scale_slice_xy_enabled)
							compute_desc_.contrast_min_slice_xy = value;
						else
							compute_desc_.contrast_min_slice_xy = pow(10, value);
					}
					else if (compute_desc_.current_window == WindowKind::XZview)
					{
						if (compute_desc_.log_scale_slice_xz_enabled)
							compute_desc_.contrast_min_slice_xz = value;
						else
							compute_desc_.contrast_min_slice_xz = pow(10, value);
					}
					else if (compute_desc_.current_window == WindowKind::YZview)
					{
						if (compute_desc_.log_scale_slice_yz_enabled)
							compute_desc_.contrast_min_slice_yz = value;
						else
							compute_desc_.contrast_min_slice_yz = pow(10, value);
					}
				}
				pipe_refresh();
			}
		}

		void MainWindow::set_contrast_max(const double value)
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.contrast_enabled)
				{
					if (compute_desc_.current_window == WindowKind::XYview)
					{
						if (compute_desc_.log_scale_slice_xy_enabled)
							compute_desc_.contrast_max_slice_xy = value;
						else
							compute_desc_.contrast_max_slice_xy = pow(10, value);
					}
					else if (compute_desc_.current_window == WindowKind::XZview)
					{
						if (compute_desc_.log_scale_slice_xz_enabled)
							compute_desc_.contrast_max_slice_xz = value;
						else
							compute_desc_.contrast_max_slice_xz = pow(10, value);
					}
					else if (compute_desc_.current_window == WindowKind::YZview)
					{
						if (compute_desc_.log_scale_slice_yz_enabled)
							compute_desc_.contrast_max_slice_yz = value;
						else
							compute_desc_.contrast_max_slice_yz = pow(10, value);
					}
					pipe_refresh();
				}
			}
		}

		void MainWindow::set_log_scale(const bool value)
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.current_window == WindowKind::XYview)
					compute_desc_.log_scale_slice_xy_enabled = value;
				else if (compute_desc_.current_window == WindowKind::XZview)
					compute_desc_.log_scale_slice_xz_enabled = value;
				else if (compute_desc_.current_window == WindowKind::YZview)
					compute_desc_.log_scale_slice_yz_enabled = value;
				if (compute_desc_.contrast_enabled)
				{
					set_contrast_min(ui.ContrastMinDoubleSpinBox->value());
					set_contrast_max(ui.ContrastMaxDoubleSpinBox->value());
				}
				notify();
				//set_auto_contrast();
				pipe_refresh();
			}
		}
		#pragma endregion
		/* ------------ */
		#pragma region Vibrometry
		void MainWindow::set_vibro_mode(const bool value)
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.pindex > compute_desc_.nsamples)
					compute_desc_.pindex = compute_desc_.nsamples.load();
				if (compute_desc_.vibrometry_q > compute_desc_.nsamples)
					compute_desc_.vibrometry_q = compute_desc_.nsamples.load();
				compute_desc_.vibrometry_enabled = value;
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_p_vibro(int value)
		{
			if (!is_direct_mode())
			{
				if (!compute_desc_.vibrometry_enabled)
					return;
				if (value < static_cast<int>(compute_desc_.nsamples) && value >= 0)
				{
					compute_desc_.pindex = value;
					pipe_refresh();
				}
				else
					display_error("p param has to be between 0 and n");
			}
		}

		void MainWindow::set_q_vibro(int value)
		{
			if (!is_direct_mode())
			{
				if (value < static_cast<int>(compute_desc_.nsamples) && value >= 0)
				{
					compute_desc_.vibrometry_q = value;
					pipe_refresh();
				}
				else
					display_error("q param has to be between 0 and phase #");
			}
		}
		#pragma endregion
		/* ------------ */
		#pragma region Average
		void MainWindow::set_average_mode(const bool value)
		{
			if (mainDisplay)
			{
				compute_desc_.average_enabled = value;
				mainDisplay->resetTransform();
				if (value)
					mainDisplay->getOverlayManager().create_overlay<Signal>();
				else
					mainDisplay->resetSelection();
				is_enabled_average_ = value;
				notify();
			}
		}

		void MainWindow::activeSignalZone()
		{
			mainDisplay->getOverlayManager().create_overlay<Signal>();
			notify();
		}

		void MainWindow::activeNoiseZone()
		{
			mainDisplay->getOverlayManager().create_overlay<Noise>();
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
			/* This function is used for both opening and saving a ROI file.
			   The default QFileDialog show "Open" or "Save" as accept button,
			   thus it would be confusing to the user to click on "Save" if he
			   wants to load a file.
			   So a custom QFileDialog is used where the accept button is labeled "Select"
			
			   The code below is much shorter but show the wrong label:
			   QString filename = QFileDialog::getSaveFileName(this,
			      tr("ROI output file"), "C://", tr("Ini files (*.ini)"));
				*/

			QFileDialog dialog(this);
			dialog.setFileMode(QFileDialog::AnyFile);
			dialog.setNameFilter(tr("Ini files (*.ini)"));
			dialog.setDefaultSuffix(".ini");
			dialog.setDirectory("C:\\");
			dialog.setWindowTitle("ROI output file");

			dialog.setLabelText(QFileDialog::Accept, "Select");
			if (dialog.exec()) {
				QString filename = dialog.selectedFiles()[0];

				QLineEdit* roi_output_line_edit = ui.ROIFilePathLineEdit;
				roi_output_line_edit->clear();
				roi_output_line_edit->insert(filename);
			}
			
		}

		void MainWindow::browse_roi_output_file()
		{
			QString filename = QFileDialog::getSaveFileName(this,
				tr("ROI output file"), "C://", tr("Text files (*.txt);;CSV files (*.csv)"));

			QLineEdit* roi_output_line_edit = ui.ROIOutputPathLineEdit;
			roi_output_line_edit->clear();
			roi_output_line_edit->insert(filename);
		}

		void MainWindow::save_roi()
		{
			QLineEdit* path_line_edit = ui.ROIFilePathLineEdit;
			std::string path = path_line_edit->text().toUtf8();
			if (!path.empty())
			{
				boost::property_tree::ptree ptree;
				const units::RectFd signal = mainDisplay->getSignalZone();
				const units::RectFd noise = mainDisplay->getNoiseZone();

				ptree.put("signal.top_left_x", signal.src().x());
				ptree.put("signal.top_left_y", signal.src().y());
				ptree.put("signal.bottom_right_x", signal.dst().x());
				ptree.put("signal.bottom_right_y", signal.dst().y());

				ptree.put("noise.top_left_x", noise.src().x());
				ptree.put("noise.top_left_y", noise.src().y());
				ptree.put("noise.bottom_right_x", noise.dst().x());
				ptree.put("noise.bottom_right_y", noise.dst().y());

				boost::property_tree::write_ini(path, ptree);
				display_info("Roi saved in " + path);
			}
			else
				display_error("Invalid path");
		}

		void MainWindow::load_roi()
		{
			QLineEdit* path_line_edit = ui.ROIFilePathLineEdit;
			const std::string path = path_line_edit->text().toUtf8();

			if (!path.empty())
			{
				try
				{
					boost::property_tree::ptree ptree;
					boost::property_tree::ini_parser::read_ini(path, ptree);

					units::RectFd signal;
					units::RectFd noise;
					units::ConversionData convert(mainDisplay.get());

					signal.setSrc(
						units::PointFd(convert,
							ptree.get<int>("signal.top_left_x", 0),
							ptree.get<int>("signal.top_left_y", 0)));
					signal.setDst(
						units::PointFd(convert,
							ptree.get<int>("signal.bottom_right_x", 0),
							ptree.get<int>("signal.bottom_right_y", 0)));

					noise.setSrc(
						units::PointFd(convert,
							ptree.get<int>("noise.top_left_x", 0),
							ptree.get<int>("noise.top_left_y", 0)));
					noise.setDst(
						units::PointFd(convert,
							ptree.get<int>("noise.bottom_right_x", 0),
							ptree.get<int>("noise.bottom_right_y", 0)));

					mainDisplay->setSignalZone(signal);
					mainDisplay->setNoiseZone(noise);

					mainDisplay->getOverlayManager().create_overlay<Signal>();
				}
				catch (std::exception& e)
				{
					display_error("Couldn't load ini file\n" + std::string(e.what()));
				}
			}
		}

		void MainWindow::average_record()
		{
			if (plot_window_)
			{
				plot_window_->stop_drawing();
				plot_window_.reset(nullptr);
				pipe_refresh();
			}

			QLineEdit* output_line_edit = ui.ROIOutputPathLineEdit;
			QPushButton* roi_stop_push_button = ui.ROIOutputStopPushButton;
			QSpinBox* nb_of_frames_spin_box = ui.NumberOfFramesSpinBox;

			nb_frames_ = nb_of_frames_spin_box->value();
			std::string output_path = output_line_edit->text().toUtf8();
			if (output_path == "")
			{
				roi_stop_push_button->setDisabled(true);
				return display_error("No output file");
			}

			CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
				holovibes_.get_average_queue(),
				output_path,
				nb_frames_,
				this));
			connect(CSV_record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_average_record()));
			CSV_record_thread_->start();

			roi_stop_push_button->setDisabled(false);
		}

		void MainWindow::finished_average_record()
		{
			CSV_record_thread_.reset(nullptr);
			display_info("ROI record done");

			QPushButton* roi_stop_push_button = ui.ROIOutputStopPushButton;
			roi_stop_push_button->setDisabled(true);
		}
		#pragma endregion
		/* ------------ */
		#pragma region Convolution
		void MainWindow::browse_convo_matrix_file()
		{
			QString filename = QFileDialog::getOpenFileName(this,
				tr("Matrix file"), "C://", tr("Txt files (*.txt)"));

			QLineEdit* matrix_output_line_edit = ui.ConvoMatrixPathLineEdit;
			matrix_output_line_edit->clear();
			matrix_output_line_edit->insert(filename);
		}

		void MainWindow::load_convo_matrix()
		{
			QLineEdit* path_line_edit = ui.ConvoMatrixPathLineEdit;
			const std::string path = path_line_edit->text().toUtf8();
			boost::property_tree::ptree ptree;
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
				compute_desc_.convo_matrix_width = std::stoi(matrix_size[0]);
				compute_desc_.convo_matrix_height = std::stoi(matrix_size[1]);
				compute_desc_.convo_matrix_z = std::stoi(matrix_size[2]);
				boost::trim(v_str[1]);
				boost::split(matrix, v_str[1], boost::is_any_of(delims), boost::token_compress_on);
				while (c < matrix.size())
				{
					if (matrix[c] != "")
						compute_desc_.convo_matrix.push_back(std::stof(matrix[c]));
					c++;
				}
				if ((compute_desc_.convo_matrix_width * compute_desc_.convo_matrix_height * compute_desc_.convo_matrix_z) != matrix.size())
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
		#pragma endregion
		/* ------------ */
		#pragma region Record
		void MainWindow::browse_file()
		{
			QString filename = QFileDialog::getSaveFileName(this,
				tr("Record output file"), "C://", tr("Raw files (*.raw);; All files (*)"));

			QLineEdit* path_line_edit = ui.ImageOutputPathLineEdit;
			path_line_edit->clear();
			path_line_edit->insert(filename);
		}

		std::string MainWindow::set_record_filename_properties(FrameDescriptor fd, std::string filename)
		{
			std::string slice;
			switch (compute_desc_.current_window)
			{
			case SliceXZ:
				slice = "XZ";
				break;
			case SliceYZ:
				slice = "YZ";
				break;
			default:
				slice = "XY";
				break;
			}
			std::string mode = (is_direct_mode() ? "D" : "H");

			std::string sub_str = "_" + slice
				+ "_" + mode
				+ "_" + std::to_string(fd.width)
				+ "_" + std::to_string(fd.height)
				+ "_" + std::to_string(static_cast<int>(fd.depth) << 3) + "bit"
				+ "_" + "e"; // Holovibes record only in little endian

			for (int i = static_cast<int>(filename.length()); i >= 0; --i)
				if (filename[i] == '.')
				{
					filename.insert(i, sub_str, 0, sub_str.length());
					return filename;
				}
			filename += sub_str;
			return filename;
		}

		void MainWindow::set_record()
		{
			QSpinBox*  nb_of_frames_spinbox = ui.NumberOfFramesSpinBox;
			QLineEdit* path_line_edit = ui.ImageOutputPathLineEdit;
			
			int nb_of_frames = nb_of_frames_spinbox->value();
			std::string path = path_line_edit->text().toUtf8();
			QPushButton* cancel_button = ui.ImageOutputStopPushButton;
			if (path == "")
			{
				cancel_button->setDisabled(true);
				return display_error("No output file");
			}

			Queue* queue = nullptr;
			try
			{
				queue = holovibes_.get_current_window_output_queue();
				
				if (queue)
				{
					path = set_record_filename_properties(queue->get_frame_desc(), path);
					record_thread_.reset(new ThreadRecorder(*queue, path, nb_of_frames, this));

					connect(record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_image_record()));
					record_thread_->start();

					cancel_button->setDisabled(false);
				}
				else
					throw std::exception("Unable to launch record");
			}
			catch (std::exception& e)
			{
				display_error(e.what());
			}
		}

		void MainWindow::finished_image_record()
		{
			QProgressBar* progress_bar = InfoManager::get_manager()->get_progress_bar();

			QPushButton* cancel_button = ui.ImageOutputStopPushButton;
			cancel_button->setDisabled(true);

			record_thread_.reset(nullptr);

			progress_bar->setMaximum(1);
			progress_bar->setValue(1);
			display_info("Record done");
		}
		#pragma endregion
		/* ------------ */
		#pragma region Batch

		void MainWindow::browse_batch_input()
		{
			QString filename = QFileDialog::getOpenFileName(this,
				tr("Batch input file"), "C://", tr("All files (*)"));

			QLineEdit* batch_input_line_edit = ui.BatchInputPathLineEdit;
			batch_input_line_edit->clear();
			batch_input_line_edit->insert(filename);
		}

		void MainWindow::image_batch_record()
		{
			QLineEdit* output_path = ui.ImageOutputPathLineEdit;

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

			QLineEdit* output_path = ui.ROIOutputPathLineEdit;

			is_batch_img_ = false;
			is_batch_interrupted_ = false;
			batch_record(std::string(output_path->text().toUtf8()));
		}

		void MainWindow::batch_record(const std::string& path)
		{
			if (path == "")
				return display_error("No output file");
			file_index_ = 1;
			QLineEdit* batch_input_line_edit = ui.BatchInputPathLineEdit;
			QSpinBox * frame_nb_spin_box = ui.NumberOfFramesSpinBox;

			// Getting the path to the input batch file, and the number of frames to record.
			const std::string input_path = batch_input_line_edit->text().toUtf8();
			const uint frame_nb = frame_nb_spin_box->value();
			std::string formatted_path;

			try
			{
				Queue* q = nullptr;
				
				if (compute_desc_.current_window == WindowKind::XYview)
					q = &holovibes_.get_output_queue();
				else if (compute_desc_.current_window == WindowKind::XZview)
					q = &holovibes_.get_pipe()->get_stft_slice_queue(0);
				else
					q = &holovibes_.get_pipe()->get_stft_slice_queue(1);
				// Only loading the dll at runtime
				gpib_interface_ = gpib::GpibDLL::load_gpib("gpib.dll", input_path);

				formatted_path = format_batch_output(path, file_index_);
				formatted_path = set_record_filename_properties(q->get_frame_desc(), formatted_path);

				//is_enabled_camera_ = false;

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

			QSpinBox * frame_nb_spin_box = ui.NumberOfFramesSpinBox;
			std::string path;

			if (is_batch_img_)
				path = ui.ImageOutputPathLineEdit->text().toUtf8();
			else
				path = ui.ROIOutputPathLineEdit->text().toUtf8();

			Queue *q = nullptr;

			if (compute_desc_.current_window == WindowKind::XYview)
				q = &holovibes_.get_output_queue();
			else if (compute_desc_.current_window == WindowKind::XZview)
				q = &holovibes_.get_pipe()->get_stft_slice_queue(0);
			else
				q = &holovibes_.get_pipe()->get_stft_slice_queue(1);

			std::string output_filename = format_batch_output(path, file_index_);
			output_filename = set_record_filename_properties(q->get_frame_desc(), output_filename);
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
						batch_finished_record(true);
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
			//is_enabled_camera_ = true;
			if (no_error)
				display_info("Batch record done");

			if (plot_window_)
			{
				plot_window_->stop_drawing();
				holovibes_.get_pipe()->request_average(&holovibes_.get_average_queue());
				plot_window_->start_drawing();
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

		void MainWindow::stop_image_record()
		{
			if (record_thread_)
			{
				record_thread_->stop();
				is_batch_interrupted_ = true;
			}
		}

		std::string MainWindow::format_batch_output(const std::string& path, const uint index)
		{
			std::string file_index;
			std::ostringstream convert;
			convert << std::setw(6) << std::setfill('0') << index;
			file_index = convert.str();

			std::vector<std::string> path_tokens;
			boost::split(path_tokens, path, boost::is_any_of("."));
			std::string ret = path_tokens[0] + "_" + file_index;
			if (path_tokens.size() > 1)
				ret += "." + path_tokens[1];
			return ret;
		}
		#pragma endregion
		/* ------------ */
		#pragma region Import
		void MainWindow::import_browse_file()
		{
			static QString tmp_path = "";
			QString filename = "";

			filename = QFileDialog::getOpenFileName(this,
				tr("import file"), ((tmp_path == "") ? ("C://") : (tmp_path)), tr("All files (*)"));

			QLineEdit* import_line_edit = ui.ImportPathLineEdit;

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
			compute_desc_.compute_mode = Computation::Stop;
			notify();
		}

		void MainWindow::import_file()
		{
			QLineEdit *import_line_edit = ui.ImportPathLineEdit;
			QSpinBox *width_spinbox = ui.ImportWidthSpinBox;
			QSpinBox *height_spinbox = ui.ImportHeightSpinBox;
			QSpinBox *fps_spinbox = ui.ImportFpsSpinBox;
			QSpinBox *start_spinbox = ui.ImportStartSpinBox;
			QSpinBox *end_spinbox = ui.ImportEndSpinBox;
			QComboBox *depth_spinbox = ui.ImportDepthComboBox;
			QComboBox *big_endian_checkbox = ui.ImportEndiannessComboBox;
			QCheckBox *cine = ui.CineFileCheckBox;
			QDoubleSpinBox *pixel_size_spinbox = ui.PixelSizeDoubleSpinBox;

			compute_desc_.stft_steps = std::ceil(static_cast<float>(fps_spinbox->value()) / 20.0f);
			compute_desc_.pixel_size = pixel_size_spinbox->value();
			import_file_stop();
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
			depth_multi = pow(2, depth_spinbox->currentIndex());
			FrameDescriptor frame_desc = {
				static_cast<ushort>(width_spinbox->value()),
				static_cast<ushort>(height_spinbox->value()),
				static_cast<float>(depth_multi),
				(big_endian_checkbox->currentText() == QString("Big Endian") ?
					Endianness::BigEndian : Endianness::LittleEndian) };
			is_enabled_camera_ = false;
			try
			{
				auto file_end = std::experimental::filesystem::file_size(file_src)
					/ frame_desc.frame_size();
				if (file_end > end_spinbox->value())
					file_end = end_spinbox->value();
				holovibes_.init_import_mode(
					file_src,
					frame_desc,
					true,
					fps_spinbox->value(),
					start_spinbox->value(),
					file_end,
					global::global_config.input_queue_max_size,
					holovibes_,
					ui.FileReaderProgressBar,
					this);
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
			if (depth_spinbox->currentText() == QString("16") && cine->isChecked() == false)
				big_endian_checkbox->setEnabled(true);
			QAction *settings = ui.actionSettings;
			settings->setEnabled(false);
			import_type_ = ImportType::File;
			if (holovibes_.get_tcapture() && holovibes_.get_tcapture()->stop_requested_)
			{
				import_type_ = ImportType::None;
				is_enabled_camera_ = false;
				mainDisplay.reset(nullptr);
				holovibes_.dispose_compute();
				holovibes_.dispose_capture();
			}
			notify();
		}

		void MainWindow::import_start_spinbox_update()
		{
			QSpinBox *start_spinbox = ui.ImportStartSpinBox;
			QSpinBox *end_spinbox = ui.ImportEndSpinBox;

			if (start_spinbox->value() > end_spinbox->value())
				end_spinbox->setValue(start_spinbox->value());
		}

		void MainWindow::import_end_spinbox_update()
		{
			QSpinBox *start_spinbox = ui.ImportStartSpinBox;
			QSpinBox *end_spinbox = ui.ImportEndSpinBox;

			if (end_spinbox->value() < start_spinbox->value())
				start_spinbox->setValue(end_spinbox->value());
		}

		void MainWindow::set_import_cine_file(bool value)
		{
			compute_desc_.is_cine_file = value;
			notify();
		}

		void MainWindow::seek_cine_header_data(std::string &file_src_, Holovibes& holovibes_)
		{
			QComboBox		*depth_spinbox = ui.ImportDepthComboBox;
			int				read_width = 0, read_height = 0;
			ushort			read_depth = 0;
			uint			read_pixelpermeter_x = 0, offset_to_ptr = 0;
			FILE*			file = nullptr;
			fpos_t			pos = 0;
			char			buffer[45];
			buffer[44] = 0;

			try
			{
				/*Opening file and checking if it exists*/
				fopen_s(&file, file_src_.c_str(), "rb");
				if (!file)
					throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
				std::fsetpos(file, &pos);
				/*Reading the whole cine file header*/
				if (std::fread(buffer, 1, 44, file) != 44)
					throw std::runtime_error("[READER] unable to read file: " + file_src_);
				/*Checking if the file is actually a .cine file*/
				if (std::strstr(buffer, "CI") == NULL)
					throw std::runtime_error("[READER] file " + file_src_ + " is not a valid .cine file");
				/*Reading OffImageHeader for offset to BITMAPINFOHEADER*/
				std::memcpy(&offset_to_ptr, (buffer + 24), sizeof(int));
				/*Reading value biWidth*/
				pos = offset_to_ptr + 4;
				std::fsetpos(file, &pos);
				if (std::fread(&read_width, 1, sizeof(int), file) != sizeof(int))
					throw std::runtime_error("[READER] unable to read file: " + file_src_);
				/*Reading value biHeigth*/
				pos = offset_to_ptr + 8;
				std::fsetpos(file, &pos);
				if (std::fread(&read_height, 1, sizeof(int), file) != sizeof(int))
					throw std::runtime_error("[READER] unable to read file: " + file_src_);
				/*Reading value biBitCount*/
				pos = offset_to_ptr + 14;
				std::fsetpos(file, &pos);
				if (std::fread(&read_depth, 1, sizeof(short), file) != sizeof(short))
					throw std::runtime_error("[READER] unable to read file: " + file_src_);
				/*Reading value biXpelsPerMetter*/
				pos = offset_to_ptr + 24;
				std::fsetpos(file, &pos);
				if (std::fread(&read_pixelpermeter_x, 1, sizeof(int), file) != sizeof(int))
					throw std::runtime_error("[READER] unable to read file: " + file_src_);

				/*Setting value in Qt interface*/
				depth_spinbox->setCurrentIndex((read_depth != 8));

				ui.ImportWidthSpinBox->setValue(read_width);
				read_height = std::abs(read_height);
				ui.ImportHeightSpinBox->setValue(read_height);
				compute_desc_.pixel_size = (1 / static_cast<double>(read_pixelpermeter_x)) * 1e6;
				ui.ImportEndiannessComboBox->setCurrentIndex(0); // Little Endian

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

		void MainWindow::hide_endianess()
		{
			QComboBox* depth_cbox = ui.ImportDepthComboBox;
			QString curr_value = depth_cbox->currentText();
			QComboBox* imp_cbox = ui.ImportEndiannessComboBox;

			// Changing the endianess when depth = 8 makes no sense
			imp_cbox->setEnabled(curr_value == "16");
		}

		void MainWindow::title_detect(void)
		{
			QLineEdit					*import_line_edit = ui.ImportPathLineEdit;
			QSpinBox					*import_width_box = ui.ImportWidthSpinBox;
			QSpinBox					*import_height_box = ui.ImportHeightSpinBox;
			QComboBox					*import_depth_box = ui.ImportDepthComboBox;
			QComboBox					*import_endian_box = ui.ImportEndiannessComboBox;
			const std::string			file_src = import_line_edit->text().toUtf8();
			std::vector<std::string>	strings;

			uint				width = 0;
			uint				height = 0;
			uint				depth = 0;
			bool				mode;
			bool				endian;

			boost::split(strings, file_src, boost::is_any_of("_"));
			auto size = strings.size();
			if (size < 5)
				return display_error("Title detect expect at least 5 fields separated by '_'.");

			// Mode (Direct or Hologram), unused
			auto mode_str = strings[size - 5];
			if (mode_str != "D" && mode_str != "H")
				return display_error(mode_str + " is not a supported mode.");
			mode = mode_str == "H";
			// Width
			width = std::atoi(strings[size - 4].c_str());
			// Height
			height = std::atoi(strings[size - 3].c_str());
			// Depth
			depth = std::atoi(strings[size - 2].c_str());
			if (depth != 8 && depth != 16 && depth != 32 && depth != 64)
				return display_error("The depth " + strings[size - 2] + " is not supported.");
			//Endianness
			auto endian_char = strings[size - 1][0];
			if (endian_char != 'E' && endian_char != 'e')
				return display_error("The last field must be either 'E'or 'e'.");
			endian = endian_char == 'E';

			import_width_box->setValue(width);
			import_height_box->setValue(height);
			import_depth_box->setCurrentIndex(log2(depth) - 3);
			import_endian_box->setCurrentIndex(endian);
		}
		#pragma endregion

		#pragma region Themes
		void MainWindow::set_night()
		{
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
			darkPalette.setColor(QPalette::Light, Qt::black);

			qApp->setPalette(darkPalette);
			theme_index_ = 1;
		}

		void MainWindow::set_classic()
		{
			qApp->setPalette(this->style()->standardPalette());
			qApp->setStyle(QStyleFactory::create("WindowsVista"));
			qApp->setStyleSheet("");
			theme_index_ = 0;
		}
		#pragma endregion

		#pragma region Getters
		DirectWindow *MainWindow::get_main_display()
		{
			return mainDisplay.get();
		}
		void MainWindow::update_file_reader_index(int n)
		{
			if (QThread::currentThread() != this->thread())
				emit update_file_reader_index_signal(n);
			else
				ui.FileReaderProgressBar->setValue(n);
		}
	}
}
#include "moc_MainWindow.cc"