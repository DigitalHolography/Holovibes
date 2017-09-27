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
			yzAngle(90.f),
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
			is_enabled_autofocus_(false),
			import_type_(ImportType::None),
			compute_desc_(holovibes_.get_compute_desc())
		{
			ui.setupUi(this);


			setWindowIcon(QIcon("Holovibes.ico"));
			InfoManager::get_manager(findChild<GroupBox *>("InfoGroupBox"));

			move(QPoint(532, 554));

			// Hide non default tab
			findChild<GroupBox *>("PostProcessingGroupBox")->setHidden(true);
			findChild<GroupBox *>("RecordGroupBox")->setHidden(true);
			findChild<GroupBox *>("InfoGroupBox")->setHidden(true);

			findChild<QAction *>("actionSpecial")->setChecked(false);
			findChild<QAction *>("actionRecord")->setChecked(false);
			findChild<QAction *>("actionInfo")->setChecked(false);

			layout_toggled();

			load_ini(GLOBAL_INI_PATH);

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

			QComboBox *depth_cbox = findChild<QComboBox *>("ImportDepthComboBox");
			connect(depth_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(hide_endianess()));

			QComboBox *window_cbox = findChild<QComboBox *>("WindowSelectionComboBox");
			connect(window_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(change_window()));

			resize(width(), 425);
			// Display default values
			compute_desc_.compute_mode.exchange(Computation::Direct);
			notify();
			compute_desc_.compute_mode.exchange(Computation::Stop);
			notify();
			setFocusPolicy(Qt::StrongFocus);

			// spinBox allow ',' and '.' as decimal point
			spinBoxDecimalPointReplacement(findChild<QDoubleSpinBox *>("WaveLengthDoubleSpinBox"));
			spinBoxDecimalPointReplacement(findChild<QDoubleSpinBox *>("ZDoubleSpinBox"));
			spinBoxDecimalPointReplacement(findChild<QDoubleSpinBox *>("ZStepDoubleSpinBox"));
			spinBoxDecimalPointReplacement(findChild<QDoubleSpinBox *>("PixelSizeDoubleSpinBox"));
			spinBoxDecimalPointReplacement(findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox"));
			spinBoxDecimalPointReplacement(findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox"));
			spinBoxDecimalPointReplacement(findChild<QDoubleSpinBox *>("AutofocusZMinDoubleSpinBox"));
			spinBoxDecimalPointReplacement(findChild<QDoubleSpinBox *>("AutofocusZMaxDoubleSpinBox"));
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

		
		#pragma endregion
		/* ------------ */
		#pragma region Notify
		void MainWindow::notify()
		{
			const bool is_direct = is_direct_mode();
			if (compute_desc_.compute_mode.load() == Computation::Stop)
			{
				findChild<GroupBox *>("ImageRenderingGroupBox")->setEnabled(false);
				findChild<GroupBox *>("ViewGroupBox")->setEnabled(false);
				findChild<GroupBox *>("PostProcessingGroupBox")->setEnabled(false);
				findChild<GroupBox *>("RecordGroupBox")->setEnabled(false);
				findChild<GroupBox *>("ImportGroupBox")->setEnabled(true);
				findChild<GroupBox *>("InfoGroupBox")->setEnabled(true);
				return;
			}
			else if (compute_desc_.compute_mode.load() == Computation::Direct && is_enabled_camera_)
			{
				findChild<GroupBox *>("ImageRenderingGroupBox")->setEnabled(true);
				findChild<GroupBox *>("RecordGroupBox")->setEnabled(true);
				findChild<QCheckBox *>("RecordIntegerOutputCheckBox")->setChecked(false);
				findChild<QCheckBox *>("RecordFloatOutputCheckBox")->setChecked(false);
				findChild<QCheckBox *>("RecordComplexOutputCheckBox")->setChecked(false);
				findChild<QCheckBox *>("RecordIntegerOutputCheckBox")->setEnabled(false);
			}
			else if (compute_desc_.compute_mode.load() == Computation::Hologram && is_enabled_camera_)
			{
				findChild<GroupBox *>("ImageRenderingGroupBox")->setEnabled(true);
				findChild<GroupBox *>("ViewGroupBox")->setEnabled(true);
				findChild<GroupBox *>("PostProcessingGroupBox")->setEnabled(true);
				findChild<GroupBox *>("RecordGroupBox")->setEnabled(true);
				findChild<QCheckBox *>("RecordIntegerOutputCheckBox")->setChecked(true);
				findChild<QCheckBox *>("RecordIntegerOutputCheckBox")->setEnabled(true);
			}
			findChild<QCheckBox *>("RecordFloatOutputCheckBox")->setEnabled(!is_direct);
			findChild<QCheckBox *>("RecordComplexOutputCheckBox")->setEnabled(!is_direct);

			findChild<QLineEdit *>("ROIOutputPathLineEdit")->setEnabled(!is_direct && compute_desc_.average_enabled.load());
			findChild<QToolButton *>("ROIOutputToolButton")->setEnabled(!is_direct && compute_desc_.average_enabled.load());
			findChild<QPushButton *>("ROIOutputRecPushButton")->setEnabled(!is_direct && compute_desc_.average_enabled.load());
			findChild<QPushButton *>("ROIOutputBatchPushButton")->setEnabled(!is_direct && compute_desc_.average_enabled.load());
			findChild<QPushButton *>("ROIOutputStopPushButton")->setEnabled(!is_direct && compute_desc_.average_enabled.load());
			findChild<QToolButton *>("ROIFileBrowseToolButton")->setEnabled(compute_desc_.average_enabled.load());
			findChild<QLineEdit *>("ROIFilePathLineEdit")->setEnabled(compute_desc_.average_enabled.load());
			findChild<QPushButton *>("SaveROIPushButton")->setEnabled(compute_desc_.average_enabled.load());
			findChild<QPushButton *>("LoadROIPushButton")->setEnabled(compute_desc_.average_enabled.load());

			QPushButton* signalBtn = findChild<QPushButton *>("AverageSignalPushButton");
			signalBtn->setEnabled(compute_desc_.average_enabled.load());
			signalBtn->setStyleSheet((signalBtn->isEnabled() &&
				mainDisplay->getKindOfOverlay() == KindOfOverlay::Signal) ? "QPushButton {color: #8E66D9;}" : "");

			QPushButton* noiseBtn = findChild<QPushButton *>("AverageNoisePushButton");
			noiseBtn->setEnabled(compute_desc_.average_enabled.load());
			noiseBtn->setStyleSheet((noiseBtn->isEnabled() &&
				mainDisplay->getKindOfOverlay() == KindOfOverlay::Noise) ? "QPushButton {color: #00A4AB;}" : "");

			findChild<QCheckBox*>("PhaseUnwrap2DCheckBox")->
				setEnabled(((!is_direct && (compute_desc_.img_type.load() == ImgType::PhaseIncrease) ||
				(compute_desc_.img_type.load() == ImgType::Argument)) ? (true) : (false)));

			findChild<QCheckBox *>("STFTCutsCheckBox")->setEnabled(!is_direct && compute_desc_.stft_enabled.load()
				&& !compute_desc_.filter_2d_enabled.load() && !compute_desc_.vision_3d_enabled.load());
			findChild<QCheckBox *>("STFTCutsCheckBox")->setChecked(!is_direct && compute_desc_.stft_view_enabled.load());

			QPushButton *filter_button = findChild<QPushButton *>("Filter2DPushButton");
			filter_button->setEnabled(!is_direct && !compute_desc_.stft_view_enabled.load()
				&& !compute_desc_.filter_2d_enabled.load() && !compute_desc_.stft_view_enabled.load() && !compute_desc_.vision_3d_enabled.load());
			filter_button->setStyleSheet((!is_direct && compute_desc_.filter_2d_enabled.load()) ? "QPushButton {color: #009FFF;}" : "");
			findChild<QPushButton *>("CancelFilter2DPushButton")->setEnabled(!is_direct && compute_desc_.filter_2d_enabled.load());

			findChild<QCheckBox *>("ContrastCheckBox")->setChecked(!is_direct && compute_desc_.contrast_enabled.load());
			findChild<QCheckBox *>("LogScaleCheckBox")->setChecked(!is_direct && compute_desc_.log_scale_slice_xy_enabled.load());
			findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")->setEnabled(!is_direct && compute_desc_.contrast_enabled.load());
			findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")->setEnabled(!is_direct && compute_desc_.contrast_enabled.load());
			findChild<QPushButton *>("AutoContrastPushButton")->setEnabled(!is_direct && compute_desc_.contrast_enabled.load());

			QComboBox *window_selection = findChild<QComboBox*>("WindowSelectionComboBox");
			window_selection->setEnabled((compute_desc_.stft_view_enabled.load()));
			window_selection->setCurrentIndex(window_selection->isEnabled() ? compute_desc_.current_window.load() : 0);

			if (compute_desc_.current_window.load() == WindowKind::XYview)
			{
				findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")
					->setValue((compute_desc_.log_scale_slice_xy_enabled.load()) ? compute_desc_.contrast_min_slice_xy.load() : log10(compute_desc_.contrast_min_slice_xy.load()));
				findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")
					->setValue((compute_desc_.log_scale_slice_xy_enabled.load()) ? compute_desc_.contrast_max_slice_xy.load() : log10(compute_desc_.contrast_max_slice_xy.load()));
				findChild<QCheckBox *>("LogScaleCheckBox")->setChecked(!is_direct && compute_desc_.log_scale_slice_xy_enabled.load());
				findChild<QCheckBox *>("ImgAccuCheckBox")->setChecked(!is_direct && compute_desc_.img_acc_slice_xy_enabled.load());
				findChild<QSpinBox *>("ImgAccuSpinBox")->setValue(compute_desc_.img_acc_slice_xy_level.load());
				findChild<QPushButton*>("RotatePushButton")->setEnabled(!compute_desc_.vision_3d_enabled.load());
				findChild<QPushButton*>("FlipPushButton")->setEnabled(!compute_desc_.vision_3d_enabled.load());
				findChild<QPushButton*>("RotatePushButton")->setText(("Rot " + std::to_string(static_cast<int>(displayAngle))).c_str());
				findChild<QPushButton*>("FlipPushButton")->setText(("Flip " + std::to_string(displayFlip)).c_str());
			}
			else if (compute_desc_.current_window.load() == WindowKind::XZview)
			{
				findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")
					->setValue((compute_desc_.log_scale_slice_xz_enabled.load()) ? compute_desc_.contrast_min_slice_xz.load() : log10(compute_desc_.contrast_min_slice_xz.load()));
				findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")
					->setValue((compute_desc_.log_scale_slice_xz_enabled.load()) ? compute_desc_.contrast_max_slice_xz.load() : log10(compute_desc_.contrast_max_slice_xz.load()));
				findChild<QCheckBox *>("LogScaleCheckBox")->setChecked(!is_direct && compute_desc_.log_scale_slice_xz_enabled.load());
				findChild<QCheckBox *>("ImgAccuCheckBox")->setChecked(!is_direct && compute_desc_.img_acc_slice_xz_enabled.load());
				findChild<QSpinBox *>("ImgAccuSpinBox")->setValue(compute_desc_.img_acc_slice_xz_level.load());
				findChild<QPushButton*>("RotatePushButton")->setText(("Rot " + std::to_string(static_cast<int>(xzAngle))).c_str());
				findChild<QPushButton*>("FlipPushButton")->setText(("Flip " + std::to_string(xzFlip)).c_str());
			}
			else if (compute_desc_.current_window.load() == WindowKind::YZview)
			{
				findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")
					->setValue((compute_desc_.log_scale_slice_yz_enabled.load()) ? compute_desc_.contrast_min_slice_yz.load() : log10(compute_desc_.contrast_min_slice_yz.load()));
				findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")
					->setValue((compute_desc_.log_scale_slice_yz_enabled.load()) ? compute_desc_.contrast_max_slice_yz.load() : log10(compute_desc_.contrast_max_slice_yz.load()));
				findChild<QCheckBox *>("LogScaleCheckBox")->setChecked(!is_direct && compute_desc_.log_scale_slice_yz_enabled.load());
				findChild<QCheckBox *>("ImgAccuCheckBox")->setChecked(!is_direct && compute_desc_.img_acc_slice_yz_enabled.load());
				findChild<QSpinBox *>("ImgAccuSpinBox")->setValue(compute_desc_.img_acc_slice_yz_level.load());
				findChild<QPushButton*>("RotatePushButton")->setText(("Rot " + std::to_string(static_cast<int>(yzAngle))).c_str());
				findChild<QPushButton*>("FlipPushButton")->setText(("Flip " + std::to_string(yzFlip)).c_str());
			}
			{
				// Modifying one of these pairs will call a signal that reads the other one from the SpinBox
				// So we need to block them to keep the values, as well as preventing too many autocontrast calls
				const QSignalBlocker blocker_xmin(findChild<QSpinBox *>("XMinAccuSpinBox"));
				const QSignalBlocker blocker_xmax(findChild<QSpinBox *>("XMaxAccuSpinBox"));
				const QSignalBlocker blocker_ymin(findChild<QSpinBox *>("YMinAccuSpinBox"));
				const QSignalBlocker blocker_ymax(findChild<QSpinBox *>("YMaxAccuSpinBox"));

				findChild<QCheckBox *>("FFTShiftCheckBox")->setChecked(compute_desc_.shift_corners_enabled.load());
				findChild<QCheckBox *>("PAccuCheckBox")->setChecked(compute_desc_.p_accu_enabled.load());
				findChild<QSpinBox *>("PMaxAccuSpinBox")->setMaximum(compute_desc_.nsamples.load());
				auto p_max = compute_desc_.p_accu_max_level.load();
				findChild<QSpinBox *>("PMinAccuSpinBox")->setValue(compute_desc_.p_accu_min_level.load());
				findChild<QSpinBox *>("PMaxAccuSpinBox")->setValue(p_max);

				findChild<QCheckBox *>("XAccuCheckBox")->setChecked(compute_desc_.x_accu_enabled.load());
				findChild<QSpinBox *>("XMinAccuSpinBox")->setMaximum(compute_desc_.x_accu_max_level.load());
				findChild<QSpinBox *>("XMaxAccuSpinBox")->setMinimum(compute_desc_.x_accu_min_level.load());
				findChild<QSpinBox *>("XMinAccuSpinBox")->setValue(compute_desc_.x_accu_min_level.load());
				findChild<QSpinBox *>("XMaxAccuSpinBox")->setValue(compute_desc_.x_accu_max_level.load());

				findChild<QCheckBox *>("YAccuCheckBox")->setChecked(compute_desc_.y_accu_enabled.load());
				findChild<QSpinBox *>("YMinAccuSpinBox")->setMaximum(compute_desc_.y_accu_max_level.load());
				findChild<QSpinBox *>("YMaxAccuSpinBox")->setMinimum(compute_desc_.y_accu_min_level.load());
				findChild<QSpinBox *>("YMinAccuSpinBox")->setValue(compute_desc_.y_accu_min_level.load());
				findChild<QSpinBox *>("YMaxAccuSpinBox")->setValue(compute_desc_.y_accu_max_level.load());

				findChild<QCheckBox *>("XAccuCheckBox")->setEnabled(!is_direct && compute_desc_.stft_view_enabled.load());
				findChild<QSpinBox *>("XMinAccuSpinBox")->setEnabled(!is_direct && compute_desc_.stft_view_enabled.load());
				findChild<QSpinBox *>("XMaxAccuSpinBox")->setEnabled(!is_direct && compute_desc_.stft_view_enabled.load());
				findChild<QCheckBox *>("YAccuCheckBox")->setEnabled(!is_direct && compute_desc_.stft_view_enabled.load());
				findChild<QSpinBox *>("YMinAccuSpinBox")->setEnabled(!is_direct && compute_desc_.stft_view_enabled.load());
				findChild<QSpinBox *>("YMaxAccuSpinBox")->setEnabled(!is_direct && compute_desc_.stft_view_enabled.load());

			}

			findChild<QCheckBox *>("PAccuCheckBox")->setEnabled(compute_desc_.stft_enabled.load());

			QSpinBox *p_vibro = findChild<QSpinBox *>("ImageRatioPSpinBox");
			p_vibro->setEnabled(!is_direct && compute_desc_.vibrometry_enabled.load());
			p_vibro->setValue(compute_desc_.pindex.load());
			p_vibro->setMaximum(compute_desc_.nsamples.load() - 1);
			QSpinBox *q_vibro = findChild<QSpinBox *>("ImageRatioQSpinBox");
			q_vibro->setEnabled(!is_direct && compute_desc_.vibrometry_enabled.load());
			q_vibro->setValue(compute_desc_.vibrometry_q.load());
			q_vibro->setMaximum(compute_desc_.nsamples.load() - 1);

			findChild<QCheckBox *>("ImageRatioCheckBox")->setChecked(!is_direct && compute_desc_.vibrometry_enabled.load());
			findChild<QCheckBox *>("ConvoCheckBox")->setEnabled(!is_direct && compute_desc_.convo_matrix.size() == 0 ? false : true);
			findChild<QCheckBox *>("AverageCheckBox")->setEnabled(!compute_desc_.stft_view_enabled.load() && !compute_desc_.vision_3d_enabled.load());
			findChild<QCheckBox *>("AverageCheckBox")->setChecked(!is_direct && compute_desc_.average_enabled.load());
			findChild<QCheckBox *>("FlowgraphyCheckBox")->setChecked(!is_direct && compute_desc_.flowgraphy_enabled.load());
			findChild<QSpinBox *>("FlowgraphyLevelSpinBox")->setEnabled(!is_direct && compute_desc_.flowgraphy_level.load());
			findChild<QSpinBox *>("FlowgraphyLevelSpinBox")->setValue(compute_desc_.flowgraphy_level.load());
			findChild<QPushButton *>("AutofocusRunPushButton")->setEnabled(!is_direct && compute_desc_.algorithm.load() != Algorithm::None && !compute_desc_.vision_3d_enabled.load());
			findChild<QLabel *>("AutofocusLabel")->setText((is_enabled_autofocus_) ? "<font color='Yellow'>Autofocus:</font>" : "Autofocus:");
			findChild<QCheckBox *>("STFTCheckBox")->setEnabled(!is_direct && !compute_desc_.stft_view_enabled.load() && !compute_desc_.vision_3d_enabled.load());
			findChild<QCheckBox *>("STFTCheckBox")->setChecked(!is_direct && compute_desc_.stft_enabled.load());
			findChild<QSpinBox *>("STFTStepsSpinBox")->setEnabled(!is_direct);
			findChild<QSpinBox *>("STFTStepsSpinBox")->setValue(compute_desc_.stft_steps.load());
			findChild<QPushButton *>("TakeRefPushButton")->setEnabled(!is_direct && !compute_desc_.ref_sliding_enabled.load());
			findChild<QPushButton *>("SlidingRefPushButton")->setEnabled(!is_direct && !compute_desc_.ref_diff_enabled.load() && !compute_desc_.ref_sliding_enabled.load());
			findChild<QPushButton *>("CancelRefPushButton")->setEnabled(!is_direct && (compute_desc_.ref_diff_enabled.load() || compute_desc_.ref_sliding_enabled.load()));
			findChild<QComboBox *>("AlgorithmComboBox")->setEnabled(!is_direct);
			findChild<QComboBox *>("AlgorithmComboBox")->setCurrentIndex(compute_desc_.algorithm.load());
			findChild<QComboBox *>("ViewModeComboBox")->setCurrentIndex(compute_desc_.img_type.load());
			findChild<QSpinBox *>("PhaseNumberSpinBox")->setEnabled(!is_direct && !compute_desc_.stft_view_enabled.load() && !compute_desc_.vision_3d_enabled.load());
			findChild<QSpinBox *>("PhaseNumberSpinBox")->setValue(compute_desc_.nsamples.load());
			findChild<QSpinBox *>("PSpinBox")->setEnabled(!is_direct && !compute_desc_.p_accu_enabled);
			findChild<QSpinBox *>("PSpinBox")->setMaximum(compute_desc_.nsamples.load() - 1);
			findChild<QSpinBox *>("PSpinBox")->setValue(compute_desc_.pindex.load());
			findChild<QDoubleSpinBox *>("WaveLengthDoubleSpinBox")->setEnabled(!is_direct);
			findChild<QDoubleSpinBox *>("WaveLengthDoubleSpinBox")->setValue(compute_desc_.lambda.load() * 1.0e9f);
			findChild<QDoubleSpinBox *>("ZDoubleSpinBox")->setEnabled(!is_direct);
			findChild<QDoubleSpinBox *>("ZDoubleSpinBox")->setValue(compute_desc_.zdistance.load());
			findChild<QDoubleSpinBox *>("ZStepDoubleSpinBox")->setEnabled(!is_direct);
			findChild<QDoubleSpinBox *>("PixelSizeDoubleSpinBox")->setEnabled(!compute_desc_.is_cine_file.load());
			findChild<QDoubleSpinBox *>("PixelSizeDoubleSpinBox")->setValue(compute_desc_.import_pixel_size.load());
			findChild<QLineEdit *>("BoundaryLineEdit")->setText(QString::number(holovibes_.get_boundary()));
			findChild<QSpinBox *>("KernelBufferSizeSpinBox")->setValue(compute_desc_.special_buffer_size.load());
			findChild<QCheckBox *>("CineFileCheckBox")->setChecked(compute_desc_.is_cine_file.load());
			findChild<QSpinBox *>("ImportWidthSpinBox")->setEnabled(!compute_desc_.is_cine_file.load());
			findChild<QSpinBox *>("ImportHeightSpinBox")->setEnabled(!compute_desc_.is_cine_file.load());
			findChild<QComboBox *>("ImportDepthComboBox")->setEnabled(!compute_desc_.is_cine_file.load());
			
			QString depth_value = findChild<QComboBox *>("ImportDepthComboBox")->currentText();
			findChild<QComboBox *>("ImportEndiannessComboBox")->setEnabled(depth_value == "16" && !compute_desc_.is_cine_file.load());
			
			findChild<QCheckBox *>("Vision3DCheckBox")->setEnabled(!is_direct && compute_desc_.stft_enabled.load() && !compute_desc_.stft_view_enabled.load());
			findChild<QCheckBox *>("Vision3DCheckBox")->setChecked(compute_desc_.vision_3d_enabled.load());

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
					if (compute_desc_.stft_enabled.load())
					{
						compute_desc_.pindex.exchange(0);
						compute_desc_.nsamples.exchange(1);
					}
					if (compute_desc_.flowgraphy_enabled.load() || compute_desc_.convolution_enabled.load())
					{
						compute_desc_.convolution_enabled.exchange(false);
						compute_desc_.flowgraphy_enabled.exchange(false);
						compute_desc_.special_buffer_size.exchange(3);
					}
				}
				if (err_ptr->get_kind() == error_kind::fail_accumulation)
				{
					compute_desc_.img_acc_slice_xy_enabled.exchange(false);
					compute_desc_.img_acc_slice_xy_level.exchange(1);
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
				resize(QSize(childCount * 195, 425));
			else
				resize(QSize(195, 60));
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
			save_ini("holovibes.ini");
			notify();
		}

		void MainWindow::reload_ini()
		{
			import_file_stop();
			load_ini(GLOBAL_INI_PATH);
			if (import_type_ == ImportType::File)
				import_file();
			else if (import_type_ == ImportType::Camera)
				change_camera(kCamera);
			notify();
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

			QAction	*image_rendering_action = findChild<QAction *>("actionImage_rendering");
			QAction	*view_action = findChild<QAction *>("actionView");
			QAction	*special_action = findChild<QAction *>("actionSpecial");
			QAction	*record_action = findChild<QAction *>("actionRecord");
			QAction	*import_action = findChild<QAction *>("actionImport");
			QAction	*info_action = findChild<QAction *>("actionInfo");

			try
			{
				boost::property_tree::ini_parser::read_ini(path, ptree);
			}
			catch (std::exception& e)
			{
				std::cout << e.what() << std::endl;
				return;
			}

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
				compute_desc_.special_buffer_size.exchange(ptree.get<int>("config.convolution_buffer_size", compute_desc_.special_buffer_size.load()));
				compute_desc_.stft_level.exchange(ptree.get<uint>("config.stft_buffer_size", compute_desc_.stft_level.load()));
				compute_desc_.ref_diff_level.exchange(ptree.get<uint>("config.reference_buffer_size", compute_desc_.ref_diff_level.load()));
				compute_desc_.img_acc_slice_xy_level.exchange(ptree.get<uint>("config.accumulation_buffer_size", compute_desc_.img_acc_slice_xy_level.load()));
				compute_desc_.display_rate.exchange(ptree.get<float>("config.display_rate", compute_desc_.display_rate.load()));

				// Camera type
				//const int camera_type = ptree.get<int>("image_rendering.camera", 0);
				//change_camera(static_cast<CameraKind>(camera_type));

				// Image rendering
				image_rendering_action->setChecked(!ptree.get<bool>("image_rendering.hidden", false));
				image_rendering_group_box->setHidden(ptree.get<bool>("image_rendering.hidden", false));

				const ushort p_nsample = ptree.get<ushort>("image_rendering.phase_number", compute_desc_.nsamples.load());
				if (p_nsample < 1)
					compute_desc_.nsamples.exchange(1);
				else if (p_nsample > config.input_queue_max_size)
					compute_desc_.nsamples.exchange(config.input_queue_max_size);
				else
					compute_desc_.nsamples.exchange(p_nsample);
				const ushort p_index = ptree.get<ushort>("image_rendering.p_index", compute_desc_.pindex.load());
				if (p_index >= 0 && p_index < compute_desc_.nsamples.load())
					compute_desc_.pindex.exchange(p_index);

				compute_desc_.lambda.exchange(ptree.get<float>("image_rendering.lambda", compute_desc_.lambda.load()));

				compute_desc_.zdistance.exchange(ptree.get<float>("image_rendering.z_distance", compute_desc_.zdistance.load()));

				const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
				if (z_step > 0.0f)
					z_step_ = z_step;

				compute_desc_.algorithm.exchange(static_cast<Algorithm>(ptree.get<int>("image_rendering.algorithm", compute_desc_.algorithm.load())));

				// View
				view_action->setChecked(!ptree.get<bool>("view.hidden", false));
				view_group_box->setHidden(ptree.get<bool>("view.hidden", false));

				compute_desc_.img_type.exchange(static_cast<ImgType>(
					ptree.get<int>("view.view_mode", compute_desc_.img_type.load())));
				last_img_type_ = (compute_desc_.img_type == ImgType::Complex) ?
					"Complex output" : last_img_type_;

				compute_desc_.log_scale_slice_xy_enabled.exchange(ptree.get<bool>("view.log_scale_enabled", compute_desc_.log_scale_slice_xy_enabled.load()));
				compute_desc_.log_scale_slice_xz_enabled.exchange(ptree.get<bool>("view.log_scale_enabled_cut_xz", compute_desc_.log_scale_slice_xz_enabled.load()));
				compute_desc_.log_scale_slice_yz_enabled.exchange(ptree.get<bool>("view.log_scale_enabled_cut_yz", compute_desc_.log_scale_slice_yz_enabled.load()));

				compute_desc_.shift_corners_enabled.exchange(ptree.get<bool>("view.shift_corners_enabled", compute_desc_.shift_corners_enabled.load()));

				compute_desc_.contrast_enabled.exchange(ptree.get<bool>("view.contrast_enabled", compute_desc_.contrast_enabled.load()));

				compute_desc_.contrast_min_slice_xy.exchange(ptree.get<float>("view.contrast_min", compute_desc_.contrast_min_slice_xy.load()));
				compute_desc_.contrast_max_slice_xy.exchange(ptree.get<float>("view.contrast_max", compute_desc_.contrast_max_slice_xy.load()));
				compute_desc_.cuts_contrast_p_offset.exchange(ptree.get<ushort>("view.cuts_contrast_p_offset", compute_desc_.cuts_contrast_p_offset.load()));
				if (compute_desc_.cuts_contrast_p_offset.load() < 0)
					compute_desc_.cuts_contrast_p_offset.exchange(0);
				else if (compute_desc_.cuts_contrast_p_offset.load() > compute_desc_.nsamples.load() - 1)
					compute_desc_.cuts_contrast_p_offset.exchange(compute_desc_.nsamples.load() - 1);

				compute_desc_.img_acc_slice_xy_enabled.exchange(ptree.get<bool>("view.accumulation_enabled", compute_desc_.img_acc_slice_xy_enabled.load()));

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
					ptree.get<int>("post_processing.image_ratio_q", compute_desc_.vibrometry_q.load()));
				compute_desc_.average_enabled.exchange(is_enabled_average_);

				// Record
				record_action->setChecked(!ptree.get<bool>("record.hidden", false));
				record_group_box->setHidden(ptree.get<bool>("record.hidden", false));

				// Import
				import_action->setChecked(!ptree.get<bool>("import.hidden", false));
				import_group_box->setHidden(ptree.get<bool>("import.hidden", false));
				config.import_pixel_size = ptree.get<float>("import.pixel_size", config.import_pixel_size);
				compute_desc_.import_pixel_size.exchange(config.import_pixel_size);
				findChild<QSpinBox *>("ImportFpsSpinBox")->setValue(ptree.get<int>("import.fps", 60));

				// Info
				info_action->setChecked(!ptree.get<bool>("info.hidden", false));
				info_group_box->setHidden(ptree.get<bool>("info.hidden", false));
				theme_index_ = ptree.get<int>("info.theme_type", theme_index_);

				// Autofocus
				compute_desc_.autofocus_size.exchange(ptree.get<int>("autofocus.size", compute_desc_.autofocus_size.load()));
				compute_desc_.autofocus_z_min.exchange(ptree.get<float>("autofocus.z_min", compute_desc_.autofocus_z_min.load()));
				compute_desc_.autofocus_z_max.exchange(ptree.get<float>("autofocus.z_max", compute_desc_.autofocus_z_max.load()));
				compute_desc_.autofocus_z_div.exchange(ptree.get<uint>("autofocus.steps", compute_desc_.autofocus_z_div.load()));
				compute_desc_.autofocus_z_iter.exchange(ptree.get<uint>("autofocus.loops", compute_desc_.autofocus_z_iter.load()));

				//flowgraphy
				uint flowgraphy_level = ptree.get<uint>("flowgraphy.level", compute_desc_.flowgraphy_level.load());
				compute_desc_.flowgraphy_level.exchange((flowgraphy_level % 2 == 0) ? (flowgraphy_level + 1) : (flowgraphy_level));
				compute_desc_.flowgraphy_enabled.exchange(ptree.get<bool>("flowgraphy.enable", compute_desc_.flowgraphy_enabled.load()));

				// Reset button
				config.set_cuda_device = ptree.get<bool>("reset.set_cuda_device", config.set_cuda_device);
				config.auto_device_number = ptree.get<bool>("reset.auto_device_number", config.auto_device_number);
				config.device_number = ptree.get<int>("reset.device_number", config.device_number);

				notify();
			}
		}

		void MainWindow::save_ini(const std::string& path)
		{
			boost::property_tree::ptree ptree;
			GroupBox *image_rendering_group_box = findChild<GroupBox *>("ImageRenderingGroupBox");
			GroupBox *view_group_box = findChild<GroupBox *>("ViewGroupBox");
			GroupBox *special_group_box = findChild<GroupBox *>("PostProcessingGroupBox");
			GroupBox *record_group_box = findChild<GroupBox *>("RecordGroupBox");
			GroupBox *import_group_box = findChild<GroupBox *>("ImportGroupBox");
			GroupBox *info_group_box = findChild<GroupBox *>("InfoGroupBox");
			Config& config = global::global_config;
			
			// Config
			ptree.put<uint>("config.input_buffer_size", config.input_queue_max_size);
			ptree.put<uint>("config.output_buffer_size", config.output_queue_max_size);
			ptree.put<uint>("config.float_buffer_size", config.float_queue_max_size);
			ptree.put<uint>("config.input_file_buffer_size", config.reader_buf_max_size);
			ptree.put<uint>("config.stft_cuts_output_buffer_size", config.stft_cuts_output_buffer_size);
			ptree.put<int>("config.stft_buffer_size", compute_desc_.stft_level.load());
			ptree.put<int>("config.reference_buffer_size", compute_desc_.ref_diff_level.load());
			ptree.put<uint>("config.accumulation_buffer_size", compute_desc_.img_acc_slice_xy_level.load());
			ptree.put<int>("config.convolution_buffer_size", compute_desc_.special_buffer_size.load());
			ptree.put<uint>("config.frame_timeout", config.frame_timeout);
			ptree.put<bool>("config.flush_on_refresh", config.flush_on_refresh);
			ptree.put<ushort>("config.display_rate", static_cast<ushort>(compute_desc_.display_rate.load()));

			// Image rendering
			ptree.put<bool>("image_rendering.hidden", image_rendering_group_box->isHidden());
			ptree.put("image_rendering.camera", kCamera);
			ptree.put<ushort>("image_rendering.phase_number", compute_desc_.nsamples.load());
			ptree.put<ushort>("image_rendering.p_index", compute_desc_.pindex.load());
			ptree.put<float>("image_rendering.lambda", compute_desc_.lambda.load());
			ptree.put<float>("image_rendering.z_distance", compute_desc_.zdistance.load());
			ptree.put<double>("image_rendering.z_step", z_step_);
			ptree.put<holovibes::Algorithm>("image_rendering.algorithm", compute_desc_.algorithm.load());
			
			// View
			ptree.put<bool>("view.hidden", view_group_box->isHidden());
			ptree.put<holovibes::ImgType>("view.view_mode", compute_desc_.img_type.load());
			ptree.put<bool>("view.log_scale_enabled", compute_desc_.log_scale_slice_xy_enabled.load());
			ptree.put<bool>("view.log_scale_enabled_cut_xz", compute_desc_.log_scale_slice_xz_enabled.load());
			ptree.put<bool>("view.log_scale_enabled_cut_yz", compute_desc_.log_scale_slice_yz_enabled.load());
			ptree.put<bool>("view.shift_corners_enabled", compute_desc_.shift_corners_enabled.load());
			ptree.put<bool>("view.contrast_enabled", compute_desc_.contrast_enabled.load());
			ptree.put<float>("view.contrast_min", compute_desc_.contrast_min_slice_xy.load());
			ptree.put<float>("view.contrast_max", compute_desc_.contrast_max_slice_xy.load());
			ptree.put<ushort>("view.cuts_contrast_p_offset", compute_desc_.cuts_contrast_p_offset.load());
			ptree.put<bool>("view.accumulation_enabled", compute_desc_.img_acc_slice_xy_enabled.load());
			ptree.put<float>("view.mainWindow_rotate", displayAngle);
			ptree.put<float>("view.xCut_rotate", xzAngle);
			ptree.put<float>("view.yCut_rotate", yzAngle);
			ptree.put<int>("view.mainWindow_flip", displayFlip);
			ptree.put<int>("view.xCut_flip", xzFlip);
			ptree.put<int>("view.yCut_flip", yzFlip);

			// Post-processing
			ptree.put<bool>("post_processing.hidden", special_group_box->isHidden());
			ptree.put<ushort>("post_processing.image_ratio_q", compute_desc_.vibrometry_q.load());

			// Record
			ptree.put<bool>("record.hidden", record_group_box->isHidden());

			// Import
			ptree.put<bool>("import.hidden", import_group_box->isHidden());
			ptree.put<float>("import.pixel_size", compute_desc_.import_pixel_size.load());

			// Info
			ptree.put<bool>("info.hidden", info_group_box->isHidden());
			ptree.put<ushort>("info.theme_type", theme_index_);

			// Autofocus
			ptree.put<uint>("autofocus.size", compute_desc_.autofocus_size.load());
			ptree.put<float>("autofocus.z_min", compute_desc_.autofocus_z_min.load());
			ptree.put<float>("autofocus.z_max", compute_desc_.autofocus_z_max.load());
			ptree.put<uint>("autofocus.steps", compute_desc_.autofocus_z_div.load());
			ptree.put<uint>("autofocus.loops", compute_desc_.autofocus_z_iter.load());

			//flowgraphy
			ptree.put<uint>("flowgraphy.level", compute_desc_.flowgraphy_level.load());
			ptree.put<bool>("flowgraphy.enable", compute_desc_.flowgraphy_enabled.load());

			//Reset
			ptree.put<bool>("reset.set_cuda_device", config.set_cuda_device);
			ptree.put<bool>("reset.auto_device_number", config.auto_device_number);
			ptree.put<uint>("reset.device_number", config.device_number);

			
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
			if (compute_desc_.average_enabled.load())
				set_average_mode(false);
			if (compute_desc_.vision_3d_enabled.load())
				set_vision_3d(false);
			if (compute_desc_.stft_enabled.load())
				cancel_stft_view(compute_desc_);
			if (compute_desc_.ref_diff_enabled.load() || compute_desc_.ref_sliding_enabled.load())
				cancel_take_reference();
			if (compute_desc_.filter_2d_enabled.load())
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
			findChild<QAction*>("actionSettings")->setEnabled(false);
			is_enabled_camera_ = false;
			compute_desc_.compute_mode.exchange(Computation::Stop);
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
			if (sliceXZ)
				sliceXZ.reset(nullptr);
			if (sliceYZ)
				sliceYZ.reset(nullptr);
			if (plot_window_)
				plot_window_.reset(nullptr);
			if (mainDisplay)
				mainDisplay.reset(nullptr);
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
			compute_desc_.pindex.exchange(0);
			compute_desc_.nsamples.exchange(1);
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
			load_ini(GLOBAL_INI_PATH);
			notify();
		}

		void MainWindow::closeEvent(QCloseEvent* event)
		{
			if (compute_desc_.compute_mode.load() != Computation::Stop)
				close_critical_compute();
			camera_none();
			close_windows();
			remove_infos();
			// Avoiding "unused variable" warning.
			static_cast<void*>(event);
			save_ini("holovibes.ini");
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

		void MainWindow::camera_ids()
		{
			change_camera(CameraKind::IDS);
		}

		void MainWindow::camera_ixon()
		{
			change_camera(CameraKind::Ixon);
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
			compute_desc_.compute_mode.exchange(Computation::Stop);
			notify();
			if (is_enabled_camera_)
			{
				QPoint pos(0, 0);
				QSize size(512, 512);
				init_image_mode(pos, size);
				compute_desc_.compute_mode.exchange(Computation::Direct);
				createPipe();
				mainDisplay.reset(
					new DirectWindow(
						pos, size,
						holovibes_.get_capture_queue()));
				mainDisplay->setTitle(QString("XY view"));
				mainDisplay->setCd(&compute_desc_);
				const FrameDescriptor& fd = holovibes_.get_capture_queue().get_frame_desc();
				InfoManager::insertInputSource(fd.width, fd.height, fd.depth);
				set_convolution_mode(false);
				notify();
			}
		}

		void MainWindow::createPipe()
		{
			uint depth = holovibes_.get_capture_queue().get_frame_desc().depth;
			
			if (compute_desc_.compute_mode.load() != Computation::Direct)
				compute_desc_.stft_enabled.exchange(true);
			if (compute_desc_.img_type.load() == ImgType::Complex)
				depth = 8;
			else if (compute_desc_.compute_mode.load() == Computation::Hologram)
				depth = 2;
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
						this));
				mainDisplay->setTitle(QString("XY view"));
				mainDisplay->setCd(&compute_desc_);
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
				compute_desc_.compute_mode.exchange(Computation::Hologram);
				/* ---------- */
				createPipe();
				createHoloWindow();
				/* ---------- */
				const FrameDescriptor& fd = holovibes_.get_output_queue().get_frame_desc();
				InfoManager::insertInputSource(fd.width, fd.height, fd.depth);
				/* ---------- */
				compute_desc_.contrast_enabled.exchange(true);
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

		void MainWindow::set_view_mode(const QString value)
		{
			if (!is_direct_mode())
			{
				QComboBox* ptr = findChild<QComboBox*>("ViewModeComboBox");

				compute_desc_.img_type.exchange(static_cast<ImgType>(ptr->currentIndex()));
				if ((last_img_type_ == "Complex output" && value != "Complex output") ||
					(last_img_type_ != "Complex output" && value == "Complex output"))
				{
					refreshViewMode();
					if (compute_desc_.stft_view_enabled.load())
						set_auto_contrast_cuts();
				}
				last_img_type_ = value;

				set_auto_contrast();
				notify();
			}
		}
		
		bool MainWindow::is_direct_mode()
		{
			return (compute_desc_.compute_mode.load() == Computation::Direct);
		}

		void MainWindow::set_image_mode()
		{
			if (compute_desc_.compute_mode.load() == Computation::Direct)
				set_direct_mode();
			else if (compute_desc_.compute_mode.load() == Computation::Hologram)
				set_holographic_mode();
			else
			{
				if (findChild<QRadioButton *>("DirectRadioButton")->isChecked())
					set_direct_mode();
				else
					set_holographic_mode();
			}
		}

		void MainWindow::set_vision_3d(bool checked)
		{
			const FrameDescriptor&	fd = holovibes_.get_capture_queue().get_frame_desc();

			if (checked && compute_desc_.stft_enabled.load())
			{
				QPoint pos(0, 0);
				QSize size(512, 512);
				set_average_mode(false);
				mainDisplay.reset(nullptr);
				holovibes_.get_pipe()->create_3d_vision_queue();
				while (holovibes_.get_pipe()->get_request_3d_vision());
				vision3D.reset(new Vision3DWindow(pos, size, holovibes_.get_output_queue(), compute_desc_, fd, holovibes_.get_pipe()->get_3d_vision_queue()));
				compute_desc_.vision_3d_enabled.exchange(true);
				notify();
			}
			else
			{
				compute_desc_.vision_3d_enabled.exchange(false);
				holovibes_.get_pipe()->delete_3d_vision_queue();
				while (holovibes_.get_pipe()->get_request_delete_3d_vision());
				vision3D.reset(nullptr);
				set_holographic_mode();
				notify();
			}
		}
		#pragma endregion
		/* ------------ */
		#pragma region STFT
		void MainWindow::cancel_stft_slice_view()
		{
			InfoManager *manager = InfoManager::get_manager();

			manager->remove_info("STFT Slice Cursor");

			compute_desc_.contrast_max_slice_xz.exchange(false);
			compute_desc_.contrast_max_slice_yz.exchange(false);
			compute_desc_.log_scale_slice_xz_enabled.exchange(false);
			compute_desc_.log_scale_slice_yz_enabled.exchange(false);
			compute_desc_.img_acc_slice_xz_enabled.exchange(false);
			compute_desc_.img_acc_slice_yz_enabled.exchange(false);
			holovibes_.get_pipe()->delete_stft_slice_queue();
			while (holovibes_.get_pipe()->get_cuts_delete_request());
			compute_desc_.stft_view_enabled.exchange(false);
			sliceXZ.reset(nullptr);
			sliceYZ.reset(nullptr);

			findChild<QCheckBox*>("STFTCutsCheckBox")->setChecked(false);
			findChild<QCheckBox*>("STFTCheckBox")->setEnabled(true);

			mainDisplay->setCursor(Qt::ArrowCursor);
			mainDisplay->resetSelection();
			mainDisplay->setKindOfOverlay(KindOfOverlay::Zoom);

			notify();
		}

		void MainWindow::set_stft(bool b)
		{
			if (!is_direct_mode())
			{
				try
				{
					auto NbImg_spin_box = findChild<QSpinBox *>("PhaseNumberSpinBox");
					Queue& input_queue = holovibes_.get_capture_queue();
					if (!b && static_cast<unsigned int>(NbImg_spin_box->value()) > input_queue.get_max_elts())
					{
						NbImg_spin_box->setValue(input_queue.get_max_elts());
						compute_desc_.nsamples.exchange(input_queue.get_max_elts());
					}
					compute_desc_.stft_enabled.exchange(b);
					compute_desc_.p_accu_enabled.exchange(b && findChild<QCheckBox *>("PAccuCheckBox")->isChecked());
					holovibes_.get_pipe()->request_update_n(compute_desc_.nsamples.load());
				}
				catch (std::exception& e)
				{
					compute_desc_.stft_enabled.exchange(!b);
					std::cerr << "Cannot set stft : ";
					std::cerr << e.what() << std::endl;
				}
				notify();
			}
		}

		void MainWindow::update_stft_steps(int value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.stft_steps.exchange(value);
				notify();
			}
		}

		void MainWindow::stft_view(bool checked)
		{
			InfoManager *manager = InfoManager::get_manager();
			manager->insert_info(InfoManager::InfoType::STFT_SLICE_CURSOR, "STFT Slice Cursor", "(Y,X) = (0,0)");

			QComboBox* winSelection = findChild<QComboBox*>("WindowSelectionComboBox");
			winSelection->setEnabled(checked);
			winSelection->setCurrentIndex((!checked) ? 0 : winSelection->currentIndex());
			if (checked)
			{
				try
				{
					if (compute_desc_.filter_2d_enabled.load())
						cancel_filter2D();
					holovibes_.get_pipe()->create_stft_slice_queue();
					// set positions of new windows according to the position of the main GL window
					QPoint			xzPos = mainDisplay->framePosition() + QPoint(0, mainDisplay->height() + 42);
					QPoint			yzPos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 20, 0);
					const ushort	nImg = compute_desc_.nsamples.load();
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
					sliceXZ->setPIndex(compute_desc_.pindex.load());
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
					sliceYZ->setPIndex(compute_desc_.pindex.load());
					sliceYZ->setCd(&compute_desc_);

					mainDisplay->setKindOfOverlay(KindOfOverlay::Cross);
					compute_desc_.stft_view_enabled.exchange(true);
					compute_desc_.average_enabled.exchange(false);
					set_auto_contrast_cuts();
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
			if (compute_desc_.stft_view_enabled.load())
				cancel_stft_slice_view();
			if (compute_desc_.p_accu_enabled.load())
				compute_desc_.p_accu_enabled.exchange(false);
			compute_desc_.stft_view_enabled.exchange(false);
			set_stft(false);
			notify();
		}

		#pragma endregion
		/* ------------ */
		#pragma region Computation
		void MainWindow::change_window()
		{
			QComboBox *window_cbox = findChild<QComboBox*>("WindowSelectionComboBox");

			if (window_cbox->currentIndex() == 0)
				compute_desc_.current_window.exchange(WindowKind::XYview);
			else if (window_cbox->currentIndex() == 1)
				compute_desc_.current_window.exchange(WindowKind::XZview);
			else if (window_cbox->currentIndex() == 2)
				compute_desc_.current_window.exchange(WindowKind::YZview);
			notify();
		}

		void MainWindow::set_convolution_mode(const bool value)
		{
			if (value == true && compute_desc_.convo_matrix.empty())
			{
				display_error("No valid kernel has been given");
				compute_desc_.convolution_enabled.exchange(false);
			}
			else
			{
				compute_desc_.convolution_enabled.exchange(value);
				set_auto_contrast();
			}
			notify();
		}

		void MainWindow::set_flowgraphy_mode(const bool value)
		{
			compute_desc_.flowgraphy_enabled.exchange(value);
			if (!is_direct_mode())
				pipe_refresh();
			notify();
		}

		void MainWindow::take_reference()
		{
			if (!is_direct_mode())
			{
				compute_desc_.ref_diff_enabled.exchange(true);
				holovibes_.get_pipe()->request_ref_diff_refresh();
				InfoManager::update_info("Reference", "Processing... ");
				notify();
			}
		}

		void MainWindow::take_sliding_ref()
		{
			if (!is_direct_mode())
			{
				compute_desc_.ref_sliding_enabled.exchange(true);
				holovibes_.get_pipe()->request_ref_diff_refresh();
				InfoManager::update_info("Reference", "Processing...");
				notify();
			}
		}

		void MainWindow::cancel_take_reference()
		{
			if (!is_direct_mode())
			{
				compute_desc_.ref_diff_enabled.exchange(false);
				compute_desc_.ref_sliding_enabled.exchange(false);
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
				mainDisplay->setKindOfOverlay(KindOfOverlay::Filter2D);
				findChild<QPushButton*>("Filter2DPushButton")->setStyleSheet("QPushButton {color: #009FFF;}");
				compute_desc_.log_scale_slice_xy_enabled.exchange(true);
				compute_desc_.shift_corners_enabled.exchange(false);
				compute_desc_.filter_2d_enabled.exchange(true);
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
				compute_desc_.filter_2d_enabled.exchange(false);
				compute_desc_.log_scale_slice_xy_enabled.exchange(false);
				compute_desc_.stftRoiZone(Rectangle(0, 0), AccessMode::Set);
				mainDisplay->setKindOfOverlay(KindOfOverlay::Zoom);
				mainDisplay->resetTransform();
				set_auto_contrast();
				notify();
			}
		}

		void MainWindow::set_shifted_corners(const bool value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.shift_corners_enabled.exchange(value);
				pipe_refresh();
			}
		}

		void MainWindow::setPhase()
		{
			if (!is_direct_mode())
			{
				QSpinBox *spin_box = findChild<QSpinBox *>("PhaseNumberSpinBox");
				int phaseNumber = spin_box->value();
				phaseNumber = (phaseNumber < 1) ? 1 : phaseNumber;
				Queue&				in = holovibes_.get_capture_queue();

				if (phaseNumber == compute_desc_.nsamples.load())
					return;
				if (compute_desc_.stft_enabled.load()
					|| phaseNumber <= static_cast<int>(in.get_max_elts()))
				{
					compute_desc_.nsamples.exchange(phaseNumber);
					notify();
					holovibes_.get_pipe()->request_update_n(phaseNumber);
					while (holovibes_.get_pipe()->get_request_refresh());
				}
				else
				{
					spin_box->setValue(in.get_max_elts());
					compute_desc_.nsamples.exchange(in.get_max_elts());
					notify();
					holovibes_.get_pipe()->request_update_n(in.get_max_elts());
					while (holovibes_.get_pipe()->get_request_refresh());
				}
				findChild<QSpinBox *>("PMaxAccuSpinBox")->setMaximum(compute_desc_.nsamples);
				findChild<QSpinBox *>("PMinAccuSpinBox")->setMaximum(compute_desc_.nsamples);
				set_p_accu();
			}
		}

		void MainWindow::set_special_buffer_size(int value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.special_buffer_size.exchange(value);
				if (compute_desc_.special_buffer_size.load() < static_cast<std::atomic<int>>(compute_desc_.flowgraphy_level.load()))
				{
					if (compute_desc_.special_buffer_size.load() % 2 == 0)
						compute_desc_.flowgraphy_level.exchange(compute_desc_.special_buffer_size.load() - 1);
					else
						compute_desc_.flowgraphy_level.exchange(compute_desc_.special_buffer_size.load());
				}
				notify();
				set_auto_contrast();
			}
		}

		void MainWindow::set_p_accu()
		{
			auto boxMax = findChild<QSpinBox *>("PMaxAccuSpinBox");
			auto boxMin = findChild<QSpinBox *>("PMinAccuSpinBox");
			auto checkBox = findChild<QCheckBox *>("PAccuCheckBox");
			compute_desc_.p_accu_enabled.exchange(checkBox->isChecked());
			compute_desc_.p_accu_min_level.exchange(boxMin->value());
			compute_desc_.p_accu_max_level.exchange(boxMax->value());
			if (compute_desc_.p_accu_min_level > compute_desc_.p_accu_max_level)
				boxMax->setValue(boxMin->value());

			notify();
			set_auto_contrast();
		}

		void MainWindow::set_x_accu()
		{
			compute_desc_.x_accu_enabled.exchange(findChild<QCheckBox *>("XAccuCheckBox")->isChecked());
			compute_desc_.x_accu_min_level.exchange(findChild<QSpinBox *>("XMinAccuSpinBox")->value());
			compute_desc_.x_accu_max_level.exchange(findChild<QSpinBox *>("XMaxAccuSpinBox")->value());
			set_auto_contrast();
			notify();
		}

		void MainWindow::set_y_accu()
		{
			compute_desc_.y_accu_enabled.exchange(findChild<QCheckBox *>("YAccuCheckBox")->isChecked());
			compute_desc_.y_accu_min_level.exchange(findChild<QSpinBox *>("YMinAccuSpinBox")->value());
			compute_desc_.y_accu_max_level.exchange(findChild<QSpinBox *>("YMaxAccuSpinBox")->value());
			set_auto_contrast();
			notify();
		}

		void MainWindow::set_p(int value)
		{
			if (!is_direct_mode())
			{
				if (value < static_cast<int>(compute_desc_.nsamples.load()))
				{
					compute_desc_.pindex.exchange(value);
					
					if (compute_desc_.stft_view_enabled.load())
					{
						sliceXZ->setPIndex(compute_desc_.pindex.load());
						sliceYZ->setPIndex(compute_desc_.pindex.load());
					}
					notify();
				}
				else
					display_error("p param has to be between 1 and #img");
			}
		}

		void MainWindow::set_flowgraphy_level(const int value)
		{
			int flag = 0;

			if (!is_direct_mode())
			{
				if (value % 2 == 0)
				{
					if (value + 1 <= compute_desc_.special_buffer_size.load())
					{
						compute_desc_.flowgraphy_level.exchange(value + 1);
						flag = 1;
					}
				}
				else
				{
					if (value <= compute_desc_.special_buffer_size.load())
					{
						compute_desc_.flowgraphy_level.exchange(value);
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

				if (compute_desc_.pindex.load() < compute_desc_.nsamples.load())
				{
					compute_desc_.pindex.exchange(compute_desc_.pindex.load() + 1);
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
				if (compute_desc_.pindex.load() > 0)
				{
					compute_desc_.pindex.exchange(compute_desc_.pindex.load() - 1);
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
				compute_desc_.lambda.exchange(static_cast<float>(value) * 1.0e-9f);
				pipe_refresh();
			}
		}

		void MainWindow::set_z(const double value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.zdistance.exchange(static_cast<float>(value));
				pipe_refresh();
			}
		}

		void MainWindow::increment_z()
		{
			if (!is_direct_mode())
			{
				set_z(compute_desc_.zdistance.load() + z_step_);
				findChild<QDoubleSpinBox *>("ZDoubleSpinBox")->setValue(compute_desc_.zdistance.load());
			}
		}

		void MainWindow::decrement_z()
		{
			if (!is_direct_mode())
			{
				set_z(compute_desc_.zdistance.load() - z_step_);
				findChild<QDoubleSpinBox *>("ZDoubleSpinBox")->setValue(compute_desc_.zdistance.load());
			}
		}

		void MainWindow::set_z_step(const double value)
		{
			z_step_ = value;
			findChild<QDoubleSpinBox *>("ZDoubleSpinBox")->setSingleStep(value);
		}

		void MainWindow::set_algorithm(const QString value)
		{
			if (!is_direct_mode())
			{
				if (value == "None")
					compute_desc_.algorithm.exchange(Algorithm::None);
				else if (value == "1FFT")
					compute_desc_.algorithm.exchange(Algorithm::FFT1);
				else if (value == "2FFT")
					compute_desc_.algorithm.exchange(Algorithm::FFT2);
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
				if (compute_desc_.current_window.load() == WindowKind::XYview)
					compute_desc_.img_acc_slice_xy_enabled.exchange(value);
				else if (compute_desc_.current_window.load() == WindowKind::XZview)
					compute_desc_.img_acc_slice_xz_enabled.exchange(value);
				else if (compute_desc_.current_window.load() == WindowKind::YZview)
					compute_desc_.img_acc_slice_yz_enabled.exchange(value);
				holovibes_.get_pipe()->request_acc_refresh();
				notify();
			}
		}

		void MainWindow::set_accumulation_level(int value)
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.current_window.load() == WindowKind::XYview)
					compute_desc_.img_acc_slice_xy_level.exchange(value);
				else if (compute_desc_.current_window.load() == WindowKind::XZview)
					compute_desc_.img_acc_slice_xz_level.exchange(value);
				else if (compute_desc_.current_window.load() == WindowKind::YZview)
					compute_desc_.img_acc_slice_yz_level.exchange(value);
				holovibes_.get_pipe()->request_acc_refresh();
			}
		}

		void MainWindow::set_import_pixel_size(const double value)
		{
			compute_desc_.import_pixel_size.exchange(value);
		}

		void MainWindow::set_z_iter(const int value)
		{
			if (!is_direct_mode())
				compute_desc_.autofocus_z_iter.exchange(value);
			notify();
		}

		void MainWindow::set_z_div(const int value)
		{
			if (!is_direct_mode())
				compute_desc_.autofocus_z_div.exchange(value);
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
		#pragma endregion
		/* ------------ */
		#pragma region Texture
		void MainWindow::rotateTexture()
		{
			WindowKind curWin = compute_desc_.current_window.load();

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
			WindowKind curWin = compute_desc_.current_window.load();

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
		#pragma endregion
		/* ------------ */
		#pragma region Autofocus
		void MainWindow::set_autofocus_mode()
		{
			const float	z_max = findChild<QDoubleSpinBox*>("AutofocusZMaxDoubleSpinBox")->value();
			const float	z_min = findChild<QDoubleSpinBox*>("AutofocusZMinDoubleSpinBox")->value();

			if (compute_desc_.stft_enabled.load())
				display_error("You can't call autofocus in stft mode.");
			else if (z_min < z_max)
			{
				is_enabled_autofocus_ = true;
				mainDisplay->setKindOfOverlay(KindOfOverlay::Autofocus);
				mainDisplay->resetTransform();
				InfoManager::get_manager()->update_info("Status", "Autofocus processing...");
				compute_desc_.autofocus_z_min.exchange(z_min);
				compute_desc_.autofocus_z_max.exchange(z_max);

				notify();
				is_enabled_autofocus_ = false;
			}
			else
				display_error("z min have to be strictly inferior to z max");
		}

		void MainWindow::set_z_min(const double value)
		{
			if (!is_direct_mode())
				compute_desc_.autofocus_z_min.exchange(value);
		}

		void MainWindow::set_z_max(const double value)
		{
			if (!is_direct_mode())
				compute_desc_.autofocus_z_max.exchange(value);
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
				compute_desc_.contrast_enabled.exchange(value);
				set_contrast_min(findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")->value());
				set_contrast_max(findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")->value());
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_auto_contrast_cuts()
		 {
			auto current_window = compute_desc_.current_window.load();
			compute_desc_.current_window.exchange(WindowKind::XZview);
			set_auto_contrast();
			while (holovibes_.get_pipe()->get_autocontrast_request());
			compute_desc_.current_window.exchange(WindowKind::YZview);
			set_auto_contrast();
			compute_desc_.current_window.exchange(current_window);
		}

		void MainWindow::set_auto_contrast()
		{
			if (!is_direct_mode() &&
				!compute_desc_.flowgraphy_enabled.load())
			{
				try
				{
					holovibes_.get_pipe()->request_autocontrast();
					while (holovibes_.get_pipe()->get_refresh_request());
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
				if (compute_desc_.contrast_enabled.load())
				{
					if (compute_desc_.current_window.load() == WindowKind::XYview)
					{
						if (compute_desc_.log_scale_slice_xy_enabled.load())
							compute_desc_.contrast_min_slice_xy.exchange(value);
						else
							compute_desc_.contrast_min_slice_xy.exchange(pow(10, value));
					}
					else if (compute_desc_.current_window.load() == WindowKind::XZview)
					{
						if (compute_desc_.log_scale_slice_xz_enabled.load())
							compute_desc_.contrast_min_slice_xz.exchange(value);
						else
							compute_desc_.contrast_min_slice_xz.exchange(pow(10, value));
					}
					else if (compute_desc_.current_window.load() == WindowKind::YZview)
					{
						if (compute_desc_.log_scale_slice_yz_enabled.load())
							compute_desc_.contrast_min_slice_yz.exchange(value);
						else
							compute_desc_.contrast_min_slice_yz.exchange(pow(10, value));
					}
				}
				pipe_refresh();
			}
		}

		void MainWindow::set_contrast_max(const double value)
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.contrast_enabled.load())
				{
					if (compute_desc_.current_window.load() == WindowKind::XYview)
					{
						if (compute_desc_.log_scale_slice_xy_enabled.load())
							compute_desc_.contrast_max_slice_xy.exchange(value);
						else
							compute_desc_.contrast_max_slice_xy.exchange(pow(10, value));
					}
					else if (compute_desc_.current_window.load() == WindowKind::XZview)
					{
						if (compute_desc_.log_scale_slice_xz_enabled.load())
							compute_desc_.contrast_max_slice_xz.exchange(value);
						else
							compute_desc_.contrast_max_slice_xz.exchange(pow(10, value));
					}
					else if (compute_desc_.current_window.load() == WindowKind::YZview)
					{
						if (compute_desc_.log_scale_slice_yz_enabled.load())
							compute_desc_.contrast_max_slice_yz.exchange(value);
						else
							compute_desc_.contrast_max_slice_yz.exchange(pow(10, value));
					}
					pipe_refresh();
				}
			}
		}

		void MainWindow::set_log_scale(const bool value)
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.current_window.load() == WindowKind::XYview)
					compute_desc_.log_scale_slice_xy_enabled.exchange(value);
				else if (compute_desc_.current_window.load() == WindowKind::XZview)
					compute_desc_.log_scale_slice_xz_enabled.exchange(value);
				else if (compute_desc_.current_window.load() == WindowKind::YZview)
					compute_desc_.log_scale_slice_yz_enabled.exchange(value);
				if (compute_desc_.contrast_enabled.load())
				{
					set_contrast_min(findChild<QDoubleSpinBox *>("ContrastMinDoubleSpinBox")->value());
					set_contrast_max(findChild<QDoubleSpinBox *>("ContrastMaxDoubleSpinBox")->value());
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
				if (compute_desc_.pindex.load() > compute_desc_.nsamples.load())
					compute_desc_.pindex.exchange(compute_desc_.nsamples.load());
				if (compute_desc_.vibrometry_q.load() > compute_desc_.nsamples.load())
					compute_desc_.vibrometry_q.exchange(compute_desc_.nsamples.load());
				compute_desc_.vibrometry_enabled.exchange(value);
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_p_vibro(int value)
		{
			if (!is_direct_mode())
			{
				if (!compute_desc_.vibrometry_enabled.load())
					return;
				if (value < static_cast<int>(compute_desc_.nsamples.load()) && value >= 0)
				{
					compute_desc_.pindex.exchange(value);
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
				if (value < static_cast<int>(compute_desc_.nsamples.load()) && value >= 0)
				{
					compute_desc_.vibrometry_q.exchange(value);
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
			compute_desc_.average_enabled.exchange(value);
			mainDisplay->resetTransform();
			mainDisplay->setKindOfOverlay((value) ?
				KindOfOverlay::Signal : KindOfOverlay::Zoom);
			if (!value)
				mainDisplay->resetSelection();
			is_enabled_average_ = value;
			notify();
		}

		void MainWindow::activeSignalZone()
		{
			mainDisplay->setKindOfOverlay(KindOfOverlay::Signal);
			notify();
		}

		void MainWindow::activeNoiseZone()
		{
			mainDisplay->setKindOfOverlay(KindOfOverlay::Noise);
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
			dialog.setDirectory("C:\\");
			dialog.setWindowTitle("ROI output file");

			dialog.setLabelText(QFileDialog::Accept, "Select");
			if (dialog.exec()) {
				QString filename = dialog.selectedFiles()[0];

				QLineEdit* roi_output_line_edit = findChild<QLineEdit *>("ROIFilePathLineEdit");
				roi_output_line_edit->clear();
				roi_output_line_edit->insert(filename);
			}
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
			QLineEdit* path_line_edit = findChild<QLineEdit *>("ROIFilePathLineEdit");
			std::string path = path_line_edit->text().toUtf8();
			if (!path.empty())
			{
				boost::property_tree::ptree ptree;
				const Rectangle signal = mainDisplay->getSignalZone();
				const Rectangle noise = mainDisplay->getNoiseZone();

				ptree.put("signal.top_left_x", signal.topLeft().x());
				ptree.put("signal.top_left_y", signal.topLeft().y());
				ptree.put("signal.bottom_right_x", signal.bottomRight().x());
				ptree.put("signal.bottom_right_y", signal.bottomRight().y());

				ptree.put("noise.top_left_x", noise.topLeft().x());
				ptree.put("noise.top_left_y", noise.topLeft().y());
				ptree.put("noise.bottom_right_x", noise.bottomRight().x());
				ptree.put("noise.bottom_right_y", noise.bottomRight().y());

				boost::property_tree::write_ini(path, ptree);
				display_info("Roi saved in " + path);
			}
			else
				display_error("Invalid path");
		}

		void MainWindow::load_roi()
		{
			QLineEdit* path_line_edit = findChild<QLineEdit*>("ROIFilePathLineEdit");
			const std::string path = path_line_edit->text().toUtf8();

			if (!path.empty())
			{
				try
				{
					boost::property_tree::ptree ptree;
					boost::property_tree::ini_parser::read_ini(path, ptree);

					Rectangle signal;
					Rectangle noise;

					signal.setTopLeft(
						QPoint(ptree.get<int>("signal.top_left_x", 0),
							ptree.get<int>("signal.top_left_y", 0)));
					signal.setBottomRight(
						QPoint(ptree.get<int>("signal.bottom_right_x", 0),
							ptree.get<int>("signal.bottom_right_y", 0)));

					noise.setTopLeft(
						QPoint(ptree.get<int>("noise.top_left_x", 0),
							ptree.get<int>("noise.top_left_y", 0)));
					noise.setBottomRight(
						QPoint(ptree.get<int>("noise.bottom_right_x", 0),
							ptree.get<int>("noise.bottom_right_y", 0)));

					mainDisplay->setSignalZone(signal);
					mainDisplay->setNoiseZone(noise);
					compute_desc_.signalZone(signal, AccessMode::Set);
					compute_desc_.noiseZone(noise, AccessMode::Set);

					mainDisplay->setKindOfOverlay(Signal);
				}
				catch (std::exception& e)
				{
					display_error("Couldn't load ini file\n" + std::string(e.what()));
				}
			}
		}

		void MainWindow::average_record()
		{
			/*if (plot_window_)
			{
				plot_window_->stop_drawing();
				plot_window_.reset(nullptr);
				pipe_refresh();
			}*/

			QLineEdit* output_line_edit = findChild<QLineEdit*>("ROIOutputPathLineEdit");
			std::string output_path_tmp = output_line_edit->text().toUtf8();
			if (output_path_tmp == "")
				return;

			QSpinBox* nb_of_frames_spin_box = findChild<QSpinBox*>("NumberOfFramesSpinBox");
			nb_frames_ = nb_of_frames_spin_box->value();
			Queue* q = &holovibes_.get_output_queue();
			std::string output_path = set_record_filename_properties(q->get_frame_desc(), output_path_tmp);
			CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
				holovibes_.get_average_queue(),
				output_path,
				nb_frames_,
				this));
			connect(CSV_record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_average_record()));
			CSV_record_thread_->start();

			QPushButton* roi_stop_push_button = findChild<QPushButton*>("ROIOutputStopPushButton");
			roi_stop_push_button->setDisabled(false);
		}

		void MainWindow::finished_average_record()
		{
			CSV_record_thread_.reset(nullptr);
			display_info("ROI record done");

			QPushButton* roi_stop_push_button = findChild<QPushButton*>("ROIOutputStopPushButton");
			roi_stop_push_button->setDisabled(true);
		}
		#pragma endregion
		/* ------------ */
		#pragma region Convolution
		void MainWindow::browse_convo_matrix_file()
		{
			QString filename = QFileDialog::getOpenFileName(this,
				tr("Matrix file"), "C://", tr("Txt files (*.txt)"));

			QLineEdit* matrix_output_line_edit = findChild<QLineEdit *>("ConvoMatrixPathLineEdit");
			matrix_output_line_edit->clear();
			matrix_output_line_edit->insert(filename);
		}

		void MainWindow::load_convo_matrix()
		{
			QLineEdit* path_line_edit = findChild<QLineEdit*>("ConvoMatrixPathLineEdit");
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
				return (display_error("No output file"));

			Queue* queue = nullptr;
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
				else
				{
					if (compute_desc_.current_window == WindowKind::XYview)
						queue = &holovibes_.get_output_queue();
					else if (compute_desc_.current_window == WindowKind::XZview)
						queue = &holovibes_.get_pipe()->get_stft_slice_queue(0);
					else if (compute_desc_.current_window == WindowKind::YZview)
						queue = &holovibes_.get_pipe()->get_stft_slice_queue(1);
				}
				if (queue)
				{
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
		#pragma endregion
		/* ------------ */
		#pragma region Batch

		void MainWindow::browse_batch_input()
		{
			QString filename = QFileDialog::getOpenFileName(this,
				tr("Batch input file"), "C://", tr("All files (*)"));

			QLineEdit* batch_input_line_edit = findChild<QLineEdit*>("BatchInputPathLineEdit");
			batch_input_line_edit->clear();
			batch_input_line_edit->insert(filename);
		}

		void MainWindow::set_float_visible(bool value)
		{
			QCheckBox* complex_checkbox = findChild<QCheckBox*>("RecordComplexOutputCheckBox");
			if (complex_checkbox->isChecked() && value == true)
				complex_checkbox->setChecked(false);
			QCheckBox* integer_checkbox = findChild<QCheckBox*>("RecordIntegerOutputCheckBox");
			if (integer_checkbox->isChecked() && value == true)
				integer_checkbox->setChecked(false);
		}
		
		void MainWindow::set_complex_visible(bool value)
		{
			QCheckBox* float_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");
			if (float_checkbox->isChecked() && value == true)
				float_checkbox->setChecked(false);
			QCheckBox* integer_checkbox = findChild<QCheckBox*>("RecordIntegerOutputCheckBox");
			if (integer_checkbox->isChecked() && value == true)
				integer_checkbox->setChecked(false);
		}

		void MainWindow::set_integer_visible(bool value)
		{
			if (is_direct_mode())
				findChild<QCheckBox*>("RecordIntegerOutputCheckBox")->setChecked(true);
			QCheckBox* float_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");
			if (float_checkbox->isChecked() && value == true)
				float_checkbox->setChecked(false);
			QCheckBox* complex_checkbox = findChild<QCheckBox*>("RecordComplexOutputCheckBox");
			if (complex_checkbox->isChecked() && value == true)
				complex_checkbox->setChecked(false);
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
			/*if (plot_window_)
			{
				plot_window_->stop_drawing();
				plot_window_.reset(nullptr);
				pipe_refresh();
			}*/

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
			std::string formatted_path;

			try
			{
				Queue* q = nullptr;
				
				if (compute_desc_.current_window == WindowKind::XYview)
					q = &holovibes_.get_output_queue();
				else if (compute_desc_.current_window == WindowKind::XZview)
					q = &holovibes_.get_pipe()->get_stft_slice_queue(0);
				else if (compute_desc_.current_window == WindowKind::YZview)
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

			QSpinBox * frame_nb_spin_box = findChild<QSpinBox*>("NumberOfFramesSpinBox");
			std::string path;

			if (is_batch_img_)
				path = findChild<QLineEdit*>("ImageOutputPathLineEdit")->text().toUtf8();
			else
				path = findChild<QLineEdit*>("ROIOutputPathLineEdit")->text().toUtf8();

			Queue *q = nullptr;

			if (compute_desc_.current_window == WindowKind::XYview)
				q = &holovibes_.get_output_queue();
			else if (compute_desc_.current_window == WindowKind::XZview)
				q = &holovibes_.get_pipe()->get_stft_slice_queue(0);
			else if (compute_desc_.current_window == WindowKind::YZview)
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
		#pragma endregion
		/* ------------ */
		#pragma region Import
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
			compute_desc_.compute_mode.exchange(Computation::Stop);
			notify();
		}

		void MainWindow::import_file()
		{
			import_file_stop();
			QLineEdit *import_line_edit = findChild<QLineEdit *>("ImportPathLineEdit");
			QSpinBox *width_spinbox = findChild<QSpinBox *>("ImportWidthSpinBox");
			QSpinBox *height_spinbox = findChild<QSpinBox *>("ImportHeightSpinBox");
			QSpinBox *fps_spinbox = findChild<QSpinBox *>("ImportFpsSpinBox");
			QSpinBox *start_spinbox = findChild<QSpinBox *>("ImportStartSpinBox");
			QSpinBox *end_spinbox = findChild<QSpinBox *>("ImportEndSpinBox");
			QComboBox *depth_spinbox = findChild<QComboBox *>("ImportDepthComboBox");
			QComboBox *big_endian_checkbox = findChild<QComboBox *>("ImportEndiannessComboBox");
			QCheckBox *cine = findChild<QCheckBox *>("CineFileCheckBox");
			compute_desc_.stft_steps.exchange(std::ceil(static_cast<float>(fps_spinbox->value()) / 20.0f));
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
				static_cast<float>(compute_desc_.import_pixel_size.load()),
				(big_endian_checkbox->currentText() == QString("Big Endian") ?
					Endianness::BigEndian : Endianness::LittleEndian) };
			is_enabled_camera_ = false;
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
			if (depth_spinbox->currentText() == QString("16") && cine->isChecked() == false)
				big_endian_checkbox->setEnabled(true);
			QAction *settings = findChild<QAction*>("actionSettings");
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

		void MainWindow::set_import_cine_file(bool value)
		{
			compute_desc_.is_cine_file.exchange(value);
			notify();
		}

		void MainWindow::seek_cine_header_data(std::string &file_src_, Holovibes& holovibes_)
		{
			QComboBox		*depth_spinbox = findChild<QComboBox*>("ImportDepthComboBox");
			int				read_width = 0, read_height = 0;
			ushort			read_depth = 0;
			uint			read_pixelpermeter_x = 0, offset_to_ptr = 0;
			FILE*			file = nullptr;
			fpos_t			pos = 0;
			size_t			length = 0;
			char			buffer[44];

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
				read_height = std::abs(read_height);
				findChild<QSpinBox*>("ImportHeightSpinBox")->setValue(read_height);
				compute_desc_.import_pixel_size.exchange((1 / static_cast<double>(read_pixelpermeter_x)) * 1e6);
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

		void MainWindow::hide_endianess()
		{
			QComboBox* depth_cbox = findChild<QComboBox*>("ImportDepthComboBox");
			QString curr_value = depth_cbox->currentText();
			QComboBox* imp_cbox = findChild<QComboBox*>("ImportEndiannessComboBox");

			// Changing the endianess when depth = 8 makes no sense
			imp_cbox->setEnabled(curr_value == "16");
		}

		void MainWindow::title_detect(void)
		{
			QLineEdit			*import_line_edit = findChild<QLineEdit*>("ImportPathLineEdit");
			QSpinBox			*import_width_box = findChild<QSpinBox*>("ImportWidthSpinBox");
			QSpinBox			*import_height_box = findChild<QSpinBox*>("ImportHeightSpinBox");
			QComboBox			*import_depth_box = findChild<QComboBox*>("ImportDepthComboBox");
			QComboBox			*import_endian_box = findChild<QComboBox*>("ImportEndiannessComboBox");
			const std::string	file_src = import_line_edit->text().toUtf8();
			std::string			err_msg = "Cannot detect title properties";
			uint				width = 0, height = 0, depth = 0, underscore = 5;
			size_t				i;
			bool				mode, endian;

			for (i = file_src.length(); i > 0 && underscore; --i)
				if (file_src[i] == '_')
					underscore--;
			if (underscore)
				return (display_error(err_msg));
			if (file_src[++i] == '_' && i++)
				if (file_src[i] == 'D' || file_src[i] == 'H')
					mode = ((file_src[i] == 'D') ? (false) : (true));
				else
					return (display_error(err_msg));
			if (file_src[++i] == '_')
			{
				width = std::atoi(&file_src[++i]);
				while (file_src[i] != '_' && file_src[i])
					++i;
			}
			else
				return (display_error(err_msg));
			if (file_src[i++] == '_')
			{
				height = std::atoi(&file_src[i++]);
				while (file_src[i] != '_' && file_src[i])
					++i;
			}
			else
				return (display_error(err_msg));
			if (file_src[i++] == '_')
			{
				depth = std::atoi(&file_src[i++]);
				while (file_src[i] != '_' && file_src[i])
					++i;
			}
			else
				return (display_error(err_msg));
			if (file_src[i++] == '_')
			{
				if (file_src[i] == 'e' || file_src[i] == 'E')
					endian = ((file_src[i] == 'e') ? (false) : (true));
				else
					return (display_error(err_msg));
			}
			if (depth != 8 && depth != 16 && depth != 32 && depth != 64)
				return (display_error(err_msg));

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

			qApp->setPalette(darkPalette);

			qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
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
	}
}
