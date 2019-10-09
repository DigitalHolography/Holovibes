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

#include <filesystem>

#include <QAction>
#include <QDesktopServices>
#include <QFileDialog>
#include <QMessageBox>
#include <QRect>
#include <QScreen>
#include <QShortcut>
#include <QStyleFactory>

#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "ui_mainwindow.h"
#include "MainWindow.hh"
#include "pipe.hh"
#include "logger.hh"
#include "holo_file.hh"

#define MIN_IMG_NB_STFT_CUTS 8

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
			z_step_(0.005f),
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

			qRegisterMetaType<std::function<void()>>();
			connect(this, SIGNAL(synchronize_thread_signal(std::function<void()>)), this, SLOT(synchronize_thread(std::function<void()>)));

			setWindowIcon(QIcon("Holovibes.ico"));
			InfoManager::get_manager(ui.InfoGroupBox);

			QRect rec = QGuiApplication::primaryScreen()->geometry();
			int screen_height = rec.height();
			int screen_width = rec.width();

			// need the correct dimensions of main windows
			move(QPoint((screen_width - 800) / 2, (screen_height - 500) / 2));
			show();

			// Hide non default tab
			ui.CompositeGroupBox->setHidden(true);
			ui.actionSpecial->setChecked(false);
			ui.actionMotionFocus->setChecked(false);

			try
			{
				load_ini(GLOBAL_INI_PATH);
			}
			catch (std::exception&)
			{
				LOG_WARN(std::string(GLOBAL_INI_PATH) + ": Configuration file not found. Initialization with default values.");
				save_ini(GLOBAL_INI_PATH);
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

			close_windows();
			close_critical_compute();
			camera_none();
			remove_infos();

			holovibes_.dispose_compute();
			if (!is_direct_mode())
				holovibes_.dispose_capture();
			InfoManager::get_manager()->stop_display();
		}


#pragma endregion
		/* ------------ */
#pragma region Notify
		void MainWindow::synchronize_thread(std::function<void()> f)
		{
			// We can't update gui values from a different thread
			// so we pass it to the right on using a signal
			// (This whole notify thing needs to be cleaned up / removed)
			if (QThread::currentThread() != this->thread())
				emit synchronize_thread_signal(f);
			else
				f();
		}
		void MainWindow::notify()
		{
			synchronize_thread([this]() {on_notify(); });
		}
		void MainWindow::on_notify()
		{
			const bool is_direct = is_direct_mode();

			// Tabs
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
				ui.MotionFocusGroupBox->setEnabled(true);
			}

			// Record
			ui.RawRecordingCheckBox->setEnabled(!is_direct);
			ui.SynchronizedRecordCheckBox->setEnabled(import_type_ == File);

			// Average ROI recording
			ui.RoiOutputGroupBox->setEnabled(compute_desc_.average_enabled);

			// Average
			ui.AverageGroupBox->setChecked(!is_direct && compute_desc_.average_enabled);

			QPushButton* signalBtn = ui.AverageSignalPushButton;
			signalBtn->setStyleSheet((signalBtn->isEnabled() &&
				mainDisplay && mainDisplay->getKindOfOverlay() == KindOfOverlay::Signal) ? "QPushButton {color: #8E66D9;}" : "");

			QPushButton* noiseBtn = ui.AverageNoisePushButton;
			noiseBtn->setStyleSheet((noiseBtn->isEnabled() &&
				mainDisplay && mainDisplay->getKindOfOverlay() == KindOfOverlay::Noise) ? "QPushButton {color: #00A4AB;}" : "");

			// Autofocus
			ui.AutofocusGroupBox->setEnabled(compute_desc_.algorithm != Algorithm::None);
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

			// Jitter correction

			ui.JitterCheckBox->setEnabled(compute_desc_.croped_stft);

			// Displaying mode
			ui.ViewModeComboBox->setCurrentIndex(compute_desc_.img_type);

			ui.PhaseUnwrap2DCheckBox->
				setEnabled(compute_desc_.img_type == ImgType::PhaseIncrease ||
					compute_desc_.img_type == ImgType::Argument);

			// STFT cuts
			ui.squarePixel_checkBox->setEnabled(ui.STFTCutsCheckBox->isChecked());
			ui.STFTCutsCheckBox->setChecked(!is_direct && compute_desc_.stft_view_enabled);

			// Contrast
			ui.ContrastCheckBox->setChecked(!is_direct && compute_desc_.contrast_enabled);
			ui.ContrastCheckBox->setEnabled(true);

			// FFT shift
			ui.FFTShiftCheckBox->setChecked(compute_desc_.shift_corners_enabled);
			ui.FFTShiftCheckBox->setEnabled(true);

			// Window selection
			QComboBox *window_selection = ui.WindowSelectionComboBox;
			window_selection->setEnabled(compute_desc_.stft_view_enabled);
			window_selection->setCurrentIndex(window_selection->isEnabled() ? compute_desc_.current_window : 0);

			ui.ContrastMinDoubleSpinBox->setValue(compute_desc_.get_contrast_min(compute_desc_.current_window));
			ui.ContrastMaxDoubleSpinBox->setValue(compute_desc_.get_contrast_max(compute_desc_.current_window));
			ui.LogScaleCheckBox->setEnabled(true);
			ui.LogScaleCheckBox->setChecked(!is_direct && compute_desc_.get_img_log_scale_slice_enabled(compute_desc_.current_window));
			ui.ImgAccuCheckBox->setEnabled(true);
			ui.ImgAccuCheckBox->setChecked(!is_direct && compute_desc_.get_img_acc_slice_enabled(compute_desc_.current_window));
			ui.ImgAccuSpinBox->setValue(compute_desc_.get_img_acc_slice_level(compute_desc_.current_window));
			if (compute_desc_.current_window == WindowKind::XYview)
			{
				ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(displayAngle))).c_str());
				ui.FlipPushButton->setText(("Flip " + std::to_string(displayFlip)).c_str());
			}
			else if (compute_desc_.current_window == WindowKind::XZview)
			{
				ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(xzAngle))).c_str());
				ui.FlipPushButton->setText(("Flip " + std::to_string(xzFlip)).c_str());
			}
			else if (compute_desc_.current_window == WindowKind::YZview)
			{
				ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(yzAngle))).c_str());
				ui.FlipPushButton->setText(("Flip " + std::to_string(yzFlip)).c_str());
			}

			// p accu
			ui.PAccuCheckBox->setEnabled(compute_desc_.img_type != PhaseIncrease);
			ui.PAccuCheckBox->setChecked(compute_desc_.p_accu_enabled);
			ui.PAccSpinBox->setMaximum(compute_desc_.nSize - 1);
			if (compute_desc_.p_acc_level > compute_desc_.nSize - 1)
				compute_desc_.p_acc_level = compute_desc_.nSize - 1;
			ui.PAccSpinBox->setValue(compute_desc_.p_acc_level);
			ui.PAccSpinBox->setEnabled(compute_desc_.img_type != PhaseIncrease);
			if (compute_desc_.p_accu_enabled)
			{
				ui.PSpinBox->setMaximum(compute_desc_.nSize - compute_desc_.p_acc_level - 1);
				if (compute_desc_.pindex > compute_desc_.nSize - compute_desc_.p_acc_level - 1)
					compute_desc_.pindex = compute_desc_.nSize - compute_desc_.p_acc_level - 1;
				ui.PSpinBox->setValue(compute_desc_.pindex);
				ui.PAccSpinBox->setMaximum(compute_desc_.nSize - compute_desc_.pindex - 1);
			}
			else
			{
				ui.PSpinBox->setMaximum(compute_desc_.nSize - 1);
				if (compute_desc_.pindex > compute_desc_.nSize - 1)
					compute_desc_.pindex = compute_desc_.nSize - 1;
				ui.PSpinBox->setValue(compute_desc_.pindex);
			}

			// XY accu
			ui.XAccuCheckBox->setChecked(compute_desc_.x_accu_enabled);
			ui.XAccSpinBox->setValue(compute_desc_.x_acc_level);
			ui.YAccuCheckBox->setChecked(compute_desc_.y_accu_enabled);
			ui.YAccSpinBox->setValue(compute_desc_.y_acc_level);

			// Convolution buffer
			ui.KernelBufferSizeSpinBox->setValue(compute_desc_.special_buffer_size);

			// Convolution
			ui.ConvoCheckBox->setEnabled(compute_desc_.convo_matrix.size() != 0);

			// STFT
			ui.STFTStepsSpinBox->setEnabled(!is_direct);
			ui.STFTStepsSpinBox->setValue(compute_desc_.stft_steps);

			// Ref
			ui.TakeRefPushButton->setEnabled(!is_direct && !compute_desc_.ref_sliding_enabled);
			ui.SlidingRefPushButton->setEnabled(!is_direct && !compute_desc_.ref_diff_enabled && !compute_desc_.ref_sliding_enabled);
			ui.CancelRefPushButton->setEnabled(!is_direct && (compute_desc_.ref_diff_enabled || compute_desc_.ref_sliding_enabled));

			// Image rendering
			ui.AlgorithmComboBox->setEnabled(!is_direct);
			ui.AlgorithmComboBox->setCurrentIndex(compute_desc_.algorithm);
			// Changing nSize with stft cuts is supported by the pipe, but some modifications have to be done in SliceWindow, OpenGl buffers.
			ui.nSizeSpinBox->setEnabled(!is_direct && !compute_desc_.stft_view_enabled);
			ui.nSizeSpinBox->setValue(compute_desc_.nSize);
			ui.STFTCutsCheckBox->setEnabled(ui.nSizeSpinBox->value() >= MIN_IMG_NB_STFT_CUTS);

			ui.WaveLengthDoubleSpinBox->setEnabled(!is_direct);
			ui.WaveLengthDoubleSpinBox->setValue(compute_desc_.lambda * 1.0e9f);
			ui.ZDoubleSpinBox->setEnabled(!is_direct);
			ui.ZDoubleSpinBox->setValue(compute_desc_.zdistance);
			ui.ZStepDoubleSpinBox->setEnabled(!is_direct);
			ui.BoundaryLineEdit->setText(QString::number(holovibes_.get_boundary()));

			// Filter2d
			QPushButton *filter_button = ui.Filter2DPushButton;
			filter_button->setEnabled(!is_direct && !compute_desc_.filter_2d_enabled);
			filter_button->setStyleSheet((!is_direct && compute_desc_.filter_2d_enabled) ? "QPushButton {color: #009FFF;}" : "");
			ui.CancelFilter2DPushButton->setEnabled(!is_direct && compute_desc_.filter_2d_enabled);

			// Import
			ui.CineFileCheckBox->setChecked(compute_desc_.is_cine_file);
			ui.PixelSizeDoubleSpinBox->setEnabled(!compute_desc_.is_cine_file);
			ui.PixelSizeDoubleSpinBox->setValue(compute_desc_.pixel_size);
			ui.ImportWidthSpinBox->setEnabled(!compute_desc_.is_cine_file);
			ui.ImportHeightSpinBox->setEnabled(!compute_desc_.is_cine_file);
			ui.ImportDepthComboBox->setEnabled(!compute_desc_.is_cine_file);
			QString depth_value = ui.ImportDepthComboBox->currentText();
			ui.ImportEndiannessComboBox->setEnabled(depth_value == "16" && !compute_desc_.is_cine_file);


			// Composite
			int nsize_max = compute_desc_.nSize - 1;
			ui.PRedSpinBox_Composite->setMaximum(nsize_max);
			ui.PBlueSpinBox_Composite->setMaximum(nsize_max);
			ui.SpinBox_hue_freq_min->setMaximum(nsize_max);
			ui.SpinBox_hue_freq_max->setMaximum(nsize_max);
			ui.SpinBox_saturation_freq_min->setMaximum(nsize_max);
			ui.SpinBox_saturation_freq_max->setMaximum(nsize_max);
			ui.SpinBox_value_freq_min->setMaximum(nsize_max);
			ui.SpinBox_value_freq_max->setMaximum(nsize_max);

			ui.RenormalizationCheckBox->setChecked(compute_desc_.composite_auto_weights_);

			QSpinBoxQuietSetValue(ui.PRedSpinBox_Composite, compute_desc_.composite_p_red);
			QSpinBoxQuietSetValue(ui.PBlueSpinBox_Composite, compute_desc_.composite_p_blue);
			QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_R, compute_desc_.weight_r);
			QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_G, compute_desc_.weight_g);
			QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_B, compute_desc_.weight_b);
			actualize_frequency_channel_v();

			QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_min, compute_desc_.composite_p_min_h);
			QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_max, compute_desc_.composite_p_max_h);
			QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_min, (int)(compute_desc_.slider_h_threshold_min * 1000));
			slide_update_threshold_h_min();
			QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_max, (int)(compute_desc_.slider_h_threshold_max * 1000));
			slide_update_threshold_h_max();

			QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_min, compute_desc_.composite_p_min_s);
			QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_max, compute_desc_.composite_p_max_s);
			QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_min, (int)(compute_desc_.slider_s_threshold_min * 1000));
			slide_update_threshold_s_min();
			QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_max, (int)(compute_desc_.slider_s_threshold_max * 1000));
			slide_update_threshold_s_max();

			QSpinBoxQuietSetValue(ui.SpinBox_value_freq_min, compute_desc_.composite_p_min_v);
			QSpinBoxQuietSetValue(ui.SpinBox_value_freq_max, compute_desc_.composite_p_max_v);
			QSliderQuietSetValue(ui.horizontalSlider_value_threshold_min, (int)(compute_desc_.slider_v_threshold_min * 1000));
			slide_update_threshold_v_min();
			QSliderQuietSetValue(ui.horizontalSlider_value_threshold_max, (int)(compute_desc_.slider_v_threshold_max * 1000));
			slide_update_threshold_v_max();


			ui.CompositeGroupBox->setHidden(is_direct_mode() || (compute_desc_.img_type != ImgType::Composite));

			bool rgbMode = ui.radioButton_rgb->isChecked();
			ui.groupBox->setHidden(!rgbMode);
			ui.groupBox_5->setHidden(!rgbMode && !ui.RenormalizationCheckBox->isChecked());
			ui.groupBox_hue->setHidden(rgbMode);
			ui.groupBox_saturation->setHidden(rgbMode);
			ui.groupBox_value->setHidden(rgbMode);

			// Interpolation
			ui.InterpolationCheckbox->setChecked(compute_desc_.interpolation_enabled);
			ui.InterpolationLambda1->setValue(compute_desc_.interp_lambda1 * 1.0e9f);
			ui.InterpolationLambda2->setValue(compute_desc_.interp_lambda2 * 1.0e9f);
			ui.InterpolationSensitivity->setValue(compute_desc_.interp_sensitivity);
			ui.InterpolationShift->setValue(compute_desc_.interp_shift);

			// Reticle
			if (ui.DisplayReticleCheckBox->isChecked()
				&& mainDisplay
				&& mainDisplay->getOverlayManager().getKind() != KindOfOverlay::Reticle)
			{
				ui.DisplayReticleCheckBox->setChecked(false);
				display_cross(false);
			}
		}

		void MainWindow::notify_error(std::exception& e)
		{
			CustomException* err_ptr = dynamic_cast<CustomException*>(&e);
			if (err_ptr)
			{
				if (err_ptr->get_kind() == error_kind::fail_update)
				{
					auto lambda = [this]
					{
						// notify will be in close_critical_compute
						compute_desc_.pindex = 0;
						compute_desc_.nSize = 1;
						if (compute_desc_.convolution_enabled)
						{
							compute_desc_.convolution_enabled = false;
							compute_desc_.special_buffer_size = 3;
						}
						close_windows();
						close_critical_compute();
						display_error("GPU computing error occured.\n");
						notify();
					};
					synchronize_thread(lambda);
				}
				auto lambda = [this, accu = err_ptr->get_kind() == error_kind::fail_accumulation]
				{
					if (accu)
					{
						compute_desc_.img_acc_slice_xy_enabled = false;
						compute_desc_.img_acc_slice_xy_level = 1;
					}
					close_critical_compute();

					display_error("GPU computing error occured.\n");
					notify();
				};
				synchronize_thread(lambda);
			}
			else
			{
				display_error("Unknown error occured.");
			}
		}

		void MainWindow::layout_toggled()
		{

			synchronize_thread([=]() {
				// Resizing to original size, then adjust it to fit the groupboxes
				resize(baseSize());
				adjustSize();
			});
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

				"Loïc Bellonnet-Mottet\n"
				"Antoine Martin\n"
				"François Te\n"

				"Ellena Davoine\n"
				"Clement Fang\n"
				"Danae Marmai\n"
				"Hugo Verjus\n"

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

		void MainWindow::documentation()
		{
			QMessageBox::about(0, "documentation", "<a href='https://ftp.espci.fr/incoming/Atlan/holovibes/manual/'>documentation</a>");
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
			GroupBox *motion_focus_group_box = ui.MotionFocusGroupBox;

			QAction	*image_rendering_action = ui.actionImage_rendering;
			QAction	*view_action = ui.actionView;
			QAction	*special_action = ui.actionSpecial;
			QAction	*record_action = ui.actionRecord;
			QAction	*import_action = ui.actionImport;
			QAction	*info_action = ui.actionInfo;
			QAction *motion_focus_action = ui.actionMotionFocus;

			boost::property_tree::ini_parser::read_ini(path, ptree);

			if (!ptree.empty())
			{
				Config& config = global::global_config;
				// Config
				config.input_queue_max_size = ptree.get<int>("config.input_buffer_size", config.input_queue_max_size);
				config.output_queue_max_size = ptree.get<int>("config.output_buffer_size", config.output_queue_max_size);
				config.stft_cuts_output_buffer_size = ptree.get<int>("config.stft_cuts_output_buffer_size", config.stft_cuts_output_buffer_size);
				config.frame_timeout = ptree.get<int>("config.frame_timeout", config.frame_timeout);
				config.flush_on_refresh = ptree.get<int>("config.flush_on_refresh", config.flush_on_refresh);
				config.reader_buf_max_size = ptree.get<int>("config.input_file_buffer_size", config.reader_buf_max_size);
				compute_desc_.special_buffer_size = ptree.get<int>("config.convolution_buffer_size", compute_desc_.special_buffer_size);
				compute_desc_.stft_level = ptree.get<uint>("config.stft_queue_size", compute_desc_.stft_level);
				compute_desc_.ref_diff_level = ptree.get<uint>("config.reference_buffer_size", compute_desc_.ref_diff_level);
				compute_desc_.img_acc_slice_xy_level = ptree.get<uint>("config.accumulation_buffer_size", compute_desc_.img_acc_slice_xy_level);
				compute_desc_.display_rate = ptree.get<float>("config.display_rate", compute_desc_.display_rate);

				// Image rendering
				image_rendering_action->setChecked(!ptree.get<bool>("image_rendering.hidden", image_rendering_group_box->isHidden()));

				const ushort p_nSize = ptree.get<ushort>("image_rendering.phase_number", compute_desc_.nSize);
				if (p_nSize < 1)
					compute_desc_.nSize = 1;
				else
					compute_desc_.nSize = p_nSize;
				const ushort p_index = ptree.get<ushort>("image_rendering.p_index", compute_desc_.pindex);
				if (p_index >= 0 && p_index < compute_desc_.nSize)
					compute_desc_.pindex = p_index;

				compute_desc_.lambda = ptree.get<float>("image_rendering.lambda", compute_desc_.lambda);

				compute_desc_.zdistance = ptree.get<float>("image_rendering.z_distance", compute_desc_.zdistance);

				const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
				if (z_step > 0.0f)
					set_z_step(z_step);

				compute_desc_.algorithm = static_cast<Algorithm>(ptree.get<int>("image_rendering.algorithm", compute_desc_.algorithm));

				// View
				view_action->setChecked(!ptree.get<bool>("view.hidden", view_group_box->isHidden()));

				compute_desc_.img_type.exchange(static_cast<ImgType>(
					ptree.get<int>("view.view_mode", compute_desc_.img_type)));
				last_img_type_ = compute_desc_.img_type == ImgType::Composite ? "Composite image" : last_img_type_;

				compute_desc_.log_scale_slice_xy_enabled = ptree.get<bool>("view.log_scale_enabled", compute_desc_.log_scale_slice_xy_enabled);
				compute_desc_.log_scale_slice_xz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_xz", compute_desc_.log_scale_slice_xz_enabled);
				compute_desc_.log_scale_slice_yz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_yz", compute_desc_.log_scale_slice_yz_enabled);

				compute_desc_.shift_corners_enabled = ptree.get<bool>("view.shift_corners_enabled", compute_desc_.shift_corners_enabled);

				compute_desc_.contrast_enabled = ptree.get<bool>("view.contrast_enabled", compute_desc_.contrast_enabled);
				compute_desc_.contrast_threshold_low_percentile = ptree.get<float>("view.contrast_threshold_low_percentile", compute_desc_.contrast_threshold_low_percentile);
				compute_desc_.contrast_threshold_high_percentile = ptree.get<float>("view.contrast_threshold_high_percentile", compute_desc_.contrast_threshold_high_percentile);

				compute_desc_.contrast_min_slice_xy = ptree.get<float>("view.contrast_min", compute_desc_.contrast_min_slice_xy);
				compute_desc_.contrast_max_slice_xy = ptree.get<float>("view.contrast_max", compute_desc_.contrast_max_slice_xy);
				compute_desc_.cuts_contrast_p_offset = ptree.get<ushort>("view.cuts_contrast_p_offset", compute_desc_.cuts_contrast_p_offset);
				if (compute_desc_.cuts_contrast_p_offset < 0)
					compute_desc_.cuts_contrast_p_offset = 0;
				else if (compute_desc_.cuts_contrast_p_offset > compute_desc_.nSize - 1)
					compute_desc_.cuts_contrast_p_offset = compute_desc_.nSize - 1;

				compute_desc_.img_acc_slice_xy_enabled = ptree.get<bool>("view.accumulation_enabled", compute_desc_.img_acc_slice_xy_enabled);

				displayAngle = ptree.get("view.mainWindow_rotate", displayAngle);
				xzAngle = ptree.get<float>("view.xCut_rotate", xzAngle);
				yzAngle = ptree.get<float>("view.yCut_rotate", yzAngle);
				displayFlip = ptree.get("view.mainWindow_flip", displayFlip);
				xzFlip = ptree.get("view.xCut_flip", xzFlip);
				yzFlip = ptree.get("view.yCut_flip", yzFlip);
				compute_desc_.reticle_scale = ptree.get("view.reticle_scale", 0.5f);

				// Post Processing
				special_action->setChecked(!ptree.get<bool>("post_processing.hidden", special_group_box->isHidden()));
				is_enabled_average_ = ptree.get<bool>("post_processing.average_enabled", is_enabled_average_);
				compute_desc_.average_enabled = is_enabled_average_;

				// Record
				record_action->setChecked(!ptree.get<bool>("record.hidden", record_group_box->isHidden()));

				// Motion Focus
				motion_focus_action->setChecked(!ptree.get<bool>("motion_focus.hidden", motion_focus_group_box->isHidden()));

				// Import
				import_action->setChecked(!ptree.get<bool>("import.hidden", import_group_box->isHidden()));
				compute_desc_.pixel_size = ptree.get<float>("import.pixel_size", compute_desc_.pixel_size);
				ui.ImportFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));

				// Info
				info_action->setChecked(!ptree.get<bool>("info.hidden", info_group_box->isHidden()));
				theme_index_ = ptree.get<int>("info.theme_type", theme_index_);

				// Autofocus
				compute_desc_.autofocus_size = ptree.get<int>("autofocus.size", compute_desc_.autofocus_size);
				compute_desc_.autofocus_z_min = ptree.get<float>("autofocus.z_min", compute_desc_.autofocus_z_min);
				compute_desc_.autofocus_z_max = ptree.get<float>("autofocus.z_max", compute_desc_.autofocus_z_max);
				compute_desc_.autofocus_z_div = ptree.get<uint>("autofocus.steps", compute_desc_.autofocus_z_div);
				compute_desc_.autofocus_z_iter = ptree.get<uint>("autofocus.loops", compute_desc_.autofocus_z_iter);

				// Reset button
				config.set_cuda_device = ptree.get<bool>("reset.set_cuda_device", config.set_cuda_device);
				config.auto_device_number = ptree.get<bool>("reset.auto_device_number", config.auto_device_number);
				config.device_number = ptree.get<int>("reset.device_number", config.device_number);

				// Composite
				compute_desc_.composite_p_red = ptree.get<ushort>("composite.p_red", 1);
				compute_desc_.composite_p_blue = ptree.get<ushort>("composite.p_blue", 1);
				compute_desc_.weight_r = ptree.get<float>("composite.weight_r", 1);
				compute_desc_.weight_g = ptree.get<float>("composite.weight_g", 1);
				compute_desc_.weight_b = ptree.get<float>("composite.weight_b", 1);

				compute_desc_.composite_p_min_h = ptree.get<ushort>("composite.p_min_h", 1);
				compute_desc_.composite_p_max_h = ptree.get<ushort>("composite.p_max_h", 1);
				compute_desc_.slider_h_threshold_min = ptree.get<float>("composite.slider_h_threshold_min", 0);
				compute_desc_.slider_h_threshold_max = ptree.get<float>("composite.slider_h_threshold_max", 1.0f);
				compute_desc_.composite_low_h_threshold = ptree.get<float>("composite.low_h_threshold", 0.2f);
				compute_desc_.composite_high_h_threshold = ptree.get<float>("composite.high_h_threshold", 99.8f);

				compute_desc_.composite_p_activated_s = ptree.get<bool>("composite.p_activated_s", false);
				compute_desc_.composite_p_min_s = ptree.get<ushort>("composite.p_min_s", 1);
				compute_desc_.composite_p_max_s = ptree.get<ushort>("composite.p_max_s", 1);
				compute_desc_.slider_s_threshold_min = ptree.get<float>("composite.slider_s_threshold_min", 0);
				compute_desc_.slider_s_threshold_max = ptree.get<float>("composite.slider_s_threshold_max", 1.0f);
				compute_desc_.composite_low_s_threshold = ptree.get<float>("composite.low_s_threshold", 0.2f);
				compute_desc_.composite_high_s_threshold = ptree.get<float>("composite.high_s_threshold", 99.8f);

				compute_desc_.composite_p_activated_v = ptree.get<bool>("composite.p_activated_v", false);
				compute_desc_.composite_p_min_v = ptree.get<ushort>("composite.p_min_v", 1);
				compute_desc_.composite_p_max_v = ptree.get<ushort>("composite.p_max_v", 1);
				compute_desc_.slider_v_threshold_min = ptree.get<float>("composite.slider_v_threshold_min", 0);
				compute_desc_.slider_v_threshold_max = ptree.get<float>("composite.slider_v_threshold_max", 1.0f);
				compute_desc_.composite_low_v_threshold = ptree.get<float>("composite.low_v_threshold", 0.2f);
				compute_desc_.composite_high_v_threshold = ptree.get<float>("composite.high_v_threshold", 99.8f);

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
			GroupBox *motion_focus_group_box = ui.MotionFocusGroupBox;
			Config& config = global::global_config;

			// Config
			ptree.put<uint>("config.input_buffer_size", config.input_queue_max_size);
			ptree.put<uint>("config.output_buffer_size", config.output_queue_max_size);
			ptree.put<uint>("config.input_file_buffer_size", config.reader_buf_max_size);
			ptree.put<uint>("config.stft_cuts_output_buffer_size", config.stft_cuts_output_buffer_size);
			ptree.put<int>("config.stft_queue_size", compute_desc_.stft_level);
			ptree.put<int>("config.reference_buffer_size", compute_desc_.ref_diff_level);
			ptree.put<uint>("config.accumulation_buffer_size", compute_desc_.img_acc_slice_xy_level);
			ptree.put<int>("config.convolution_buffer_size", compute_desc_.special_buffer_size);
			ptree.put<uint>("config.frame_timeout", config.frame_timeout);
			ptree.put<bool>("config.flush_on_refresh", config.flush_on_refresh);
			ptree.put<ushort>("config.display_rate", static_cast<ushort>(compute_desc_.display_rate));

			// Image rendering
			ptree.put<bool>("image_rendering.hidden", image_rendering_group_box->isHidden());
			ptree.put("image_rendering.camera", kCamera);
			ptree.put<ushort>("image_rendering.phase_number", compute_desc_.nSize);
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
			ptree.put<float>("view.contrast_threshold_low_percentile", compute_desc_.contrast_threshold_low_percentile);
			ptree.put<float>("view.contrast_threshold_high_percentile", compute_desc_.contrast_threshold_high_percentile);

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
			ptree.put<float>("view.reticle_scale", compute_desc_.reticle_scale);

			// Post-processing
			ptree.put<bool>("post_processing.hidden", special_group_box->isHidden());
			ptree.put<bool>("post_processing.average_enabled", is_enabled_average_);

			// Record
			ptree.put<bool>("record.hidden", record_group_box->isHidden());

			// Motion Focus
			ptree.put<bool>("motion_focus.hidden", motion_focus_group_box->isHidden());

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
			ptree.put<ushort>("composite.p_red", compute_desc_.composite_p_red);
			ptree.put<ushort>("composite.p_blue", compute_desc_.composite_p_blue);
			ptree.put<float>("composite.weight_r", compute_desc_.weight_r);
			ptree.put<float>("composite.weight_g", compute_desc_.weight_g);
			ptree.put<float>("composite.weight_b", compute_desc_.weight_b);

			ptree.put<ushort>("composite.p_min_h", compute_desc_.composite_p_min_h);
			ptree.put<ushort>("composite.p_max_h", compute_desc_.composite_p_max_h);
			ptree.put<float>("composite.slider_h_threshold_min", compute_desc_.slider_h_threshold_min);
			ptree.put<float>("composite.slider_h_threshold_max", compute_desc_.slider_h_threshold_max);
			ptree.put<float>("composite.low_h_threshold", compute_desc_.composite_low_h_threshold);
			ptree.put<float>("composite.high_h_threshold", compute_desc_.composite_high_h_threshold);

			ptree.put<bool>("composite.p_activated_s", compute_desc_.composite_p_activated_s);
			ptree.put<ushort>("composite.p_min_s", compute_desc_.composite_p_min_s);
			ptree.put<ushort>("composite.p_max_s", compute_desc_.composite_p_max_s);
			ptree.put<float>("composite.slider_s_threshold_min", compute_desc_.slider_s_threshold_min);
			ptree.put<float>("composite.slider_s_threshold_max", compute_desc_.slider_s_threshold_max);
			ptree.put<float>("composite.low_s_threshold", compute_desc_.composite_low_s_threshold);
			ptree.put<float>("composite.high_s_threshold", compute_desc_.composite_high_s_threshold);

			ptree.put<bool>("composite.p_activated_v", compute_desc_.composite_p_activated_v);
			ptree.put<ushort>("composite.p_min_v", compute_desc_.composite_p_min_v);
			ptree.put<ushort>("composite.p_max_v", compute_desc_.composite_p_max_v);
			ptree.put<float>("composite.slider_v_threshold_min", compute_desc_.slider_v_threshold_min);
			ptree.put<float>("composite.slider_v_threshold_max", compute_desc_.slider_v_threshold_max);
			ptree.put<float>("composite.low_v_threshold", compute_desc_.composite_low_v_threshold);
			ptree.put<float>("composite.high_v_threshold", compute_desc_.composite_high_v_threshold);
			ptree.put<bool>("composite.auto_weights", compute_desc_.composite_auto_weights_);

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
			close_windows();
			close_critical_compute();
			if (!is_direct_mode())
				holovibes_.dispose_compute();
			holovibes_.dispose_capture();
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
			compute_desc_.nSize = 1;
			is_enabled_camera_ = false;
			if (config.set_cuda_device)
			{
				if (config.auto_device_number)
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
				LOG_WARN(std::string(GLOBAL_INI_PATH) + ": Config file not found. It will use the default values.");
			}
			notify();
		}

		void MainWindow::closeEvent(QCloseEvent*)
		{
			close_windows();
			if (compute_desc_.compute_mode != Computation::Stop)
				close_critical_compute();
			camera_none();
			remove_infos();
			save_ini(GLOBAL_INI_PATH);
		}
#pragma endregion
		/* ------------ */
#pragma region Cameras
		void MainWindow::change_camera(CameraKind c)
		{
			close_windows();
			close_critical_compute();
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
			open_file(std::filesystem::current_path().generic_string() + "/" + holovibes_.get_camera_ini_path());
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
			close_windows();
			close_critical_compute();
			InfoManager::get_manager()->remove_info("Throughput");
			compute_desc_.compute_mode = Computation::Stop;
			notify();
			if (is_enabled_camera_)
			{
				QPoint pos(0, 0);
				const FrameDescriptor& fd = holovibes_.get_capture_queue()->get_frame_desc();
				width = fd.width;
				height = fd.height;
				get_good_size(width, height, 512);
				QSize size(width, height);
				init_image_mode(pos, size);
				compute_desc_.compute_mode = Computation::Direct;
				createPipe();
				mainDisplay.reset(
					new DirectWindow(
						pos, size,
						holovibes_.get_capture_queue()));
				mainDisplay->setTitle(QString("XY view"));
				mainDisplay->setCd(&compute_desc_);
				mainDisplay->setRatio((float) ui.ImportWidthSpinBox->value() / (float) ui.ImportHeightSpinBox->value());
				InfoManager::get_manager()->insertInputSource(fd);
				set_convolution_mode(false);
				set_divide_convolution_mode(false);
				notify();
				layout_toggled();
			}
		}

		void MainWindow::createPipe()
		{

			unsigned int depth = holovibes_.get_capture_queue()->get_frame_desc().depth;

			if (compute_desc_.compute_mode == Computation::Hologram)
			{
				depth = 2;
				if (compute_desc_.img_type == ImgType::Composite)
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
				LOG_ERROR(std::string("cannot create Pipe: ") + std::string(e.what()));
			}
		}

		void MainWindow::createHoloWindow()
		{
			QPoint pos(0, 0);
			const FrameDescriptor& fd = holovibes_.get_capture_queue()->get_frame_desc();
			width = fd.width;
			height = fd.height;
			get_good_size(width, height, 512);
			QSize size(width, height);
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
				mainDisplay->set_is_resize(false);
				mainDisplay->setTitle(QString("XY view"));
				mainDisplay->setCd(&compute_desc_);
				mainDisplay->resetTransform();
				mainDisplay->setAngle(displayAngle);
				mainDisplay->setFlip(displayFlip);
				mainDisplay->setRatio((float)ui.ImportWidthSpinBox->value() / (float)ui.ImportHeightSpinBox->value());

			}
			catch (std::runtime_error& e)
			{
				LOG_ERROR(std::string("createHoloWindow: ") + std::string(e.what()));
			}
		}

		void MainWindow::set_holographic_mode()
		{
			close_windows();
			close_critical_compute();
			/* ---------- */
			try
			{
				compute_desc_.compute_mode = Computation::Hologram;
				/* ---------- */
				createPipe();
				createHoloWindow();
				/* ---------- */
				const FrameDescriptor& fd = holovibes_.get_output_queue()->get_frame_desc();
				InfoManager::get_manager()->insertInputSource(fd);
				/* ---------- */
				compute_desc_.contrast_enabled = true;
				if (!compute_desc_.is_holo_file)
				{
					set_auto_contrast();
					auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
					if (pipe)
						pipe->autocontrast_end_pipe(XYview);
				}
				ui.DivideConvoCheckBox->setEnabled(false);
				notify();
			}
			catch (std::runtime_error& e)
			{
				LOG_ERROR(std::string("cannot set holographic mode: ") + std::string(e.what()));
			}
		}

		void MainWindow::refreshViewMode()
		{
			float old_scale = 1.f;
			glm::vec2 old_translation(0.f, 0.f);
			if (mainDisplay) {
				old_scale = mainDisplay->getScale();
				old_translation = mainDisplay->getTranslate();
			}
			close_windows();
			close_critical_compute();
			try
			{
				createPipe();
				createHoloWindow();
				mainDisplay->setScale(old_scale);
				mainDisplay->setTranslate(old_translation[0], old_translation[1]);
			}
			catch (std::runtime_error& e)
			{
				mainDisplay.reset(nullptr);
				LOG_ERROR(std::string("refreshViewMode: ") + std::string(e.what()));
			}
			notify();
			layout_toggled();
		}

		namespace
		{
			// Is there a change in window pixel depth (needs to be re-opened)
			bool need_refresh(const QString& last_type, const QString& new_type)
			{
				std::vector<QString> types_needing_refresh({ "Composite image" });
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

				if (need_refresh(last_img_type_, value))
				{
				// This crash in debug mode, but surprinsingly, it works perfectly in release mode.
					compute_desc_.img_type = static_cast<ImgType>(ptr->currentIndex());
					refreshViewMode();
					if (compute_desc_.img_type == ImgType::Composite)
					{
						const unsigned min_val_composite = compute_desc_.nSize == 1 ? 0 : 1;
						const unsigned max_val_composite = compute_desc_.nSize - 1;

						ui.PRedSpinBox_Composite->setValue(min_val_composite);
						ui.SpinBox_hue_freq_min->setValue(min_val_composite);
						ui.SpinBox_saturation_freq_min->setValue(min_val_composite);
						ui.SpinBox_value_freq_min->setValue(min_val_composite);

						ui.PBlueSpinBox_Composite->setValue(max_val_composite);
						ui.SpinBox_hue_freq_max->setValue(max_val_composite);
						ui.SpinBox_saturation_freq_max->setValue(max_val_composite);
						ui.SpinBox_value_freq_max->setValue(max_val_composite);
					}
				}
				last_img_type_ = value;

				auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());

				pipe->run_end_pipe([=]() {
					compute_desc_.img_type = static_cast<ImgType>(ptr->currentIndex());
					notify();
					layout_toggled();
				});
				pipe_refresh();

				pipe->autocontrast_end_pipe(XYview);
				if (compute_desc_.stft_view_enabled)
					set_auto_contrast_cuts();
				while (pipe->get_refresh_request());
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
			sliceXZ.reset(nullptr);
			sliceYZ.reset(nullptr);

			if (mainDisplay)
			{
				mainDisplay->setCursor(Qt::ArrowCursor);
				mainDisplay->resetSelection();
			}
			if (auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get()))
			{
				pipe->run_end_pipe([=]() {
					compute_desc_.stft_view_enabled = false;
					pipe->delete_stft_slice_queue();

					ui.STFTCutsCheckBox->setChecked(false);
					notify();
				});
			}

		}

		void MainWindow::set_crop_stft(bool b)
		{
			if (!is_direct_mode())
			{
				auto lambda = [=]() {
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
					holovibes_.get_pipe()->request_update_n(compute_desc_.nSize);
					compute_desc_.croped_stft = b;
				};
				auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
				pipe->run_end_pipe(lambda);
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

			compute_desc_.square_pixel = checked && ui.squarePixel_checkBox->isChecked();

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
					const ushort	nImg = compute_desc_.nSize;
					const uint		nSize = std::max(128u, std::min(256u, (uint)nImg)) * 2;

					while (holovibes_.get_pipe()->get_update_n_request());
					while (holovibes_.get_pipe()->get_cuts_request());
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
			try {
				while (holovibes_.get_pipe()->get_refresh_request());
			}
			catch (std::exception&)
			{
			}
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
			if (value == false && compute_desc_.convolution_enabled == true)
			{
				ui.DivideConvoCheckBox->setChecked(false);
				set_divide_convolution_mode(false);
			}

			ui.DivideConvoCheckBox->setEnabled(value);
			compute_desc_.convolution_enabled = value;
			set_contrast_max(ui.ContrastMaxDoubleSpinBox->value());
			set_auto_contrast();
			notify();
		}

		void MainWindow::set_divide_convolution_mode(const bool value)
		{
			compute_desc_.divide_convolution_enabled = value;
			set_auto_contrast();
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
				if (auto pipe = dynamic_cast<Pipe*>(holovibes_.get_pipe().get()))
					pipe->autocontrast_end_pipe(XYview);
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
				int nSize = ui.nSizeSpinBox->value();
				nSize = std::max(1, nSize);

				if (nSize == compute_desc_.nSize)
					return;
				notify();
				auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
				if (pipe)
				{
					pipe->run_end_pipe([=]() {
						holovibes_.get_pipe()->request_update_n(nSize);
						compute_desc_.nSize = nSize;
						set_p_accu();
						// This will not do anything until SliceWindow::changeTexture() isn't coded.
						if (compute_desc_.stft_view_enabled)
						{
							sliceXZ->adapt();
							sliceYZ->adapt();
						}
					});
				}
			}
		}

		void MainWindow::set_special_buffer_size(int value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.special_buffer_size = value;
				set_auto_contrast();
				notify();
			}
		}

		void MainWindow::update_lens_view(bool value)
		{
			if (value)
			{
				try
				{
					// set positions of new windows according to the position of the main GL window
					QPoint			pos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 300, 0);
					auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
					if (pipe)
					{
						lens_window.reset(new DirectWindow(
							pos,
							QSize(mainDisplay->width(), mainDisplay->height()),
							pipe->get_lens_queue()));
					}
					lens_window->setTitle("Lens view");
					lens_window->setCd(&compute_desc_);
				}
				catch (std::exception& e)
				{
					std::cerr << e.what() << std::endl;
				}
			}
			else
			{
				lens_window = nullptr;
			}
			compute_desc_.gpu_lens_display_enabled = value;
			pipe_refresh();
		}

		void MainWindow::update_raw_view(bool value)
		{
			compute_desc_.raw_view = value;
			auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
			if (pipe)
				pipe->get_raw_queue()->set_display(compute_desc_.record_raw || value);
			if (value)
			{
				try
				{
					// set positions of new windows according to the position of the main GL window and Lens window
					QPoint			pos = mainDisplay->framePosition() + QPoint(mainDisplay->width() * 2 + 310, 0);
					if (pipe)
					{
						raw_window.reset(new DirectWindow(
							pos,
							QSize(mainDisplay->width(), mainDisplay->height()),
							pipe->get_raw_queue()));
					}
					raw_window->setTitle("Raw view");
					raw_window->setCd(&compute_desc_);
				}
				catch (std::exception& e)
				{
					std::cerr << e.what() << std::endl;
				}
			}
			else
			{
				raw_window = nullptr;
				if (!compute_desc_.record_raw)
					gui::InfoManager::get_manager()->remove_info("RawOutputQueue");
			}
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
				if (value < static_cast<int>(compute_desc_.nSize))
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
			ui.PRedSpinBox_Composite->setValue(std::min(ui.PRedSpinBox_Composite->value(), ui.PBlueSpinBox_Composite->value()));
			compute_desc_.composite_p_red = ui.PRedSpinBox_Composite->value();
			compute_desc_.composite_p_blue = ui.PBlueSpinBox_Composite->value();
			notify();
		}

		void MainWindow::set_composite_intervals_hsv_h_min()
		{
			compute_desc_.composite_p_min_h = ui.SpinBox_hue_freq_min->value();
		}

		void MainWindow::set_composite_intervals_hsv_h_max()
		{
			compute_desc_.composite_p_max_h = ui.SpinBox_hue_freq_max->value();
		}

		void MainWindow::set_composite_intervals_hsv_s_min()
		{
			compute_desc_.composite_p_min_s = ui.SpinBox_saturation_freq_min->value();
		}

		void MainWindow::set_composite_intervals_hsv_s_max()
		{
			compute_desc_.composite_p_max_s = ui.SpinBox_saturation_freq_max->value();
		}

		void MainWindow::set_composite_intervals_hsv_v_min()
		{
			compute_desc_.composite_p_min_v = ui.SpinBox_value_freq_min->value();
		}

		void MainWindow::set_composite_intervals_hsv_v_max()
		{
			compute_desc_.composite_p_max_v = ui.SpinBox_value_freq_max->value();
		}

		void MainWindow::set_composite_weights()
		{
			compute_desc_.weight_r = ui.WeightSpinBox_R->value();
			compute_desc_.weight_g = ui.WeightSpinBox_G->value();
			compute_desc_.weight_b = ui.WeightSpinBox_B->value();
		}


		void MainWindow::set_composite_auto_weights(bool value)
		{
			compute_desc_.composite_auto_weights_ = value;
			set_auto_contrast();
		}

		void MainWindow::click_composite_rgb_or_hsv()
		{
			compute_desc_.composite_kind = ui.radioButton_rgb->isChecked() ? CompositeKind::RGB : CompositeKind::HSV;
			if (ui.radioButton_rgb->isChecked())
			{
				ui.PRedSpinBox_Composite->setValue(ui.SpinBox_hue_freq_min->value());
				ui.PBlueSpinBox_Composite->setValue(ui.SpinBox_hue_freq_max->value());
			}
			else
			{
				ui.SpinBox_hue_freq_min->setValue(ui.PRedSpinBox_Composite->value());
				ui.SpinBox_hue_freq_max->setValue(ui.PBlueSpinBox_Composite->value());
				ui.SpinBox_saturation_freq_min->setValue(ui.PRedSpinBox_Composite->value());
				ui.SpinBox_saturation_freq_max->setValue(ui.PBlueSpinBox_Composite->value());
				ui.SpinBox_value_freq_min->setValue(ui.PRedSpinBox_Composite->value());
				ui.SpinBox_value_freq_max->setValue(ui.PBlueSpinBox_Composite->value());
			}
			notify();
		}

		void MainWindow::actualize_frequency_channel_s()
		{
			compute_desc_.composite_p_activated_s = ui.checkBox_saturation_freq->isChecked();
			ui.SpinBox_saturation_freq_min->setDisabled(!ui.checkBox_saturation_freq->isChecked());
			ui.SpinBox_saturation_freq_max->setDisabled(!ui.checkBox_saturation_freq->isChecked());
		}

		void MainWindow::actualize_frequency_channel_v()
		{
			compute_desc_.composite_p_activated_v = ui.checkBox_value_freq->isChecked();
			ui.SpinBox_value_freq_min->setDisabled(!ui.checkBox_value_freq->isChecked());
			ui.SpinBox_value_freq_max->setDisabled(!ui.checkBox_value_freq->isChecked());
		}

		void MainWindow::actualize_checkbox_h_gaussian_blur()
		{
			compute_desc_.h_blur_activated = ui.checkBox_h_gaussian_blur->isChecked();
			ui.SpinBox_hue_blur_kernel_size->setEnabled(ui.checkBox_h_gaussian_blur->isChecked());
		}

		void MainWindow::actualize_kernel_size_blur()
		{
			compute_desc_.h_blur_kernel_size = ui.SpinBox_hue_blur_kernel_size->value();
		}

		void fancy_Qslide_text_percent(char* str)
		{
			size_t len = strlen(str);
			if (len < 2)
			{
				str[1] = str[0];
				str[0] = '0';
				str[2] = '\0';
				len = 2;
			}
			str[len] = str[len - 1];
			str[len - 1] = '.';
			str[len + 1] = '%';
			str[len + 2] = '\0';
		}

		void slide_update_threshold(QSlider& slider, std::atomic<float>& receiver,
			std::atomic<float>& bound_to_update, QSlider& slider_to_update,
			QLabel& to_be_written_in, std::atomic<float>& lower_bound,
			std::atomic<float>& upper_bound)
		{
			receiver = slider.value() / 1000.0f;
			char array[10];
			sprintf_s(array, "%d", slider.value());
			fancy_Qslide_text_percent(array);
			to_be_written_in.setText(QString(array));
			if (lower_bound > upper_bound)
			{
				bound_to_update = slider.value() / 1000.0f;
				slider_to_update.setValue(slider.value());
			}
		}

		void  MainWindow::slide_update_threshold_h_min()
		{
			slide_update_threshold(*ui.horizontalSlider_hue_threshold_min, compute_desc_.slider_h_threshold_min,
				compute_desc_.slider_h_threshold_max, *ui.horizontalSlider_hue_threshold_max,
				*ui.label_hue_threshold_min, compute_desc_.slider_h_threshold_min, compute_desc_.slider_h_threshold_max);
		}

		void  MainWindow::slide_update_threshold_h_max()
		{
			slide_update_threshold(*ui.horizontalSlider_hue_threshold_max, compute_desc_.slider_h_threshold_max,
				compute_desc_.slider_h_threshold_min, *ui.horizontalSlider_hue_threshold_min,
				*ui.label_hue_threshold_max, compute_desc_.slider_h_threshold_min, compute_desc_.slider_h_threshold_max);
		}

		void MainWindow::slide_update_threshold_s_min()
		{
			slide_update_threshold(*ui.horizontalSlider_saturation_threshold_min, compute_desc_.slider_s_threshold_min,
				compute_desc_.slider_s_threshold_max, *ui.horizontalSlider_saturation_threshold_max,
				*ui.label_saturation_threshold_min, compute_desc_.slider_s_threshold_min, compute_desc_.slider_s_threshold_max);
		}

		void MainWindow::slide_update_threshold_s_max()
		{
			slide_update_threshold(*ui.horizontalSlider_saturation_threshold_max, compute_desc_.slider_s_threshold_max,
				compute_desc_.slider_s_threshold_min, *ui.horizontalSlider_saturation_threshold_min,
				*ui.label_saturation_threshold_max, compute_desc_.slider_s_threshold_min, compute_desc_.slider_s_threshold_max);
		}

		void MainWindow::slide_update_threshold_v_min()
		{
			slide_update_threshold(*ui.horizontalSlider_value_threshold_min, compute_desc_.slider_v_threshold_min,
				compute_desc_.slider_v_threshold_max, *ui.horizontalSlider_value_threshold_max,
				*ui.label_value_threshold_min, compute_desc_.slider_v_threshold_min, compute_desc_.slider_v_threshold_max);
		}

		void MainWindow::slide_update_threshold_v_max()
		{
			slide_update_threshold(*ui.horizontalSlider_value_threshold_max, compute_desc_.slider_v_threshold_max,
				compute_desc_.slider_v_threshold_min, *ui.horizontalSlider_value_threshold_min,
				*ui.label_value_threshold_max, compute_desc_.slider_v_threshold_min, compute_desc_.slider_v_threshold_max);
		}

		void MainWindow::increment_p()
		{
			if (!is_direct_mode())
			{

				if (compute_desc_.pindex < compute_desc_.nSize)
				{
					compute_desc_.pindex = compute_desc_.pindex + 1;
					set_auto_contrast();
					notify();
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
					set_auto_contrast();
					notify();
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
				{
					// Shouldn't happen
					compute_desc_.algorithm = Algorithm::None;
					LOG_ERROR("Unknown algorithm: " + value.toStdString() + ", falling back to None");
				}
				set_holographic_mode();
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
				compute_desc_.set_accumulation(compute_desc_.current_window, value);
				holovibes_.get_pipe()->request_acc_refresh();
				notify();
			}
		}

		void MainWindow::set_accumulation_level(int value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.set_accumulation_level(compute_desc_.current_window, value);
				holovibes_.get_pipe()->request_acc_refresh();
			}
		}

		void MainWindow::set_xy_stabilization_enable(bool value)
		{
			compute_desc_.xy_stabilization_enabled = value;
			pipe_refresh();
		}


		void MainWindow::pixel_size_import()
		{
			compute_desc_.pixel_size = ui.PixelSizeDoubleSpinBox->value();
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


		void MainWindow::set_aberration_enabled(bool value)
		{
			compute_desc_.aberration_enabled_ = value;
			pipe_refresh();
		}
		void MainWindow::set_aberration_factor(double value)
		{
			compute_desc_.aberration_factor_ = value;
		}
		void MainWindow::set_aberration_slices(int value)
		{
			compute_desc_.aberration_slices_ = value;
			pipe_refresh();
		}

		void MainWindow::set_jitter_enabled(bool value)
		{
			compute_desc_.jitter_enabled_ = value;
		}
		void MainWindow::set_jitter_factor(double value)
		{
			compute_desc_.jitter_factor_ = value;
		}
		void MainWindow::set_jitter_slices(int value)
		{
			compute_desc_.jitter_slices_ = value;
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
				if (mainDisplay)
					mainDisplay->getOverlayManager().create_overlay<Scale>();
				if (sliceXZ)
					sliceXZ->getOverlayManager().create_overlay<Scale>();
				if (sliceYZ)
					sliceYZ->getOverlayManager().create_overlay<Scale>();
			}
			else
			{
				if (mainDisplay)
					mainDisplay->getOverlayManager().disable_all(Scale);
				if (sliceXZ)
					sliceXZ->getOverlayManager().disable_all(Scale);
				if (sliceYZ)
					sliceYZ->getOverlayManager().disable_all(Scale);
			}

		}

		void MainWindow::set_scale_bar_correction_factor(double value)
		{
			compute_desc_.scale_bar_correction_factor = value;
		}

		void MainWindow::set_square_pixel(bool enable)
		{
			compute_desc_.square_pixel = enable;
			for (auto slice : { sliceXZ.get(), sliceYZ.get() })
				if (slice)
					slice->make_pixel_square();
		}

		void MainWindow::set_locked_zoom(bool value)
		{
			compute_desc_.locked_zoom = value;
		}

		void MainWindow::zernike_refresh()
		{
			if (std::abs(compute_desc_.zernike_m) > compute_desc_.zernike_n) {
				display_error("n has to be greater or equal to |m|");
				return;
			}
			pipe_refresh();
		}

		void MainWindow::set_zernike_enable(bool val)
		{
			compute_desc_.zernike_enabled = val;
			zernike_refresh();
		}
		void MainWindow::set_zernike_m(int m)
		{
			compute_desc_.zernike_m = m;
			zernike_refresh();
		}
		void MainWindow::set_zernike_n(int n)
		{
			compute_desc_.zernike_n = n;
			zernike_refresh();
		}

		void MainWindow::set_zernike_factor(double value)
		{
			compute_desc_.zernike_factor = value;
			zernike_refresh();
		}
#pragma endregion
		/* ------------ */
#pragma region Autofocus
		void MainWindow::set_autofocus_mode()
		{
			// If current overlay is Autofocus, disable it
			if (mainDisplay->getKindOfOverlay() == Autofocus)
			{
				mainDisplay->getOverlayManager().create_default();
				mainDisplay->getOverlayManager().disable_all(Autofocus);
				notify();
			}
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
			holovibes_.get_pipe()->request_autocontrast(XZview);
			holovibes_.get_pipe()->request_autocontrast(YZview);
			if (auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get()))
			{
				pipe->run_end_pipe([=]() {
					pipe->request_autocontrast(XZview);
					pipe->request_autocontrast(YZview);
				});
			}
		}

		void MainWindow::QSpinBoxQuietSetValue(QSpinBox * spinBox, int value)
		{
			spinBox->blockSignals(true);
			spinBox->setValue(value);
			spinBox->blockSignals(false);
		}

		void MainWindow::QSliderQuietSetValue(QSlider* slider, int value)
		{
			slider->blockSignals(true);
			slider->setValue(value);
			slider->blockSignals(false);
		}

		void MainWindow::QDoubleSpinBoxQuietSetValue(QDoubleSpinBox * spinBox, double value)
		{
			spinBox->blockSignals(true);
			spinBox->setValue(value);
			spinBox->blockSignals(false);
		}

		void MainWindow::dropEvent(QDropEvent * e)
		{
			/*auto url = e->mimeData()->urls()[0];
			auto path = url.path();
			if (path.at(0) == '/')
				path.remove(0, 1);
			ui.ImportPathLineEdit->setText(path);*/
		}

		void MainWindow::dragEnterEvent(QDragEnterEvent * e)
		{
			//if (e->mimeData()->urls()[0].fileName().endsWith(".raw"))
				//e->accept();
		}

		void MainWindow::set_auto_contrast()
		{
			if (!is_direct_mode())
			{
				try
				{
					// We need to call autocontrast *after* the pipe is refreshed for it to work
					// (Does nothing if no refresh is needed)
					while (holovibes_.get_pipe()->get_request_refresh())
						continue;

					holovibes_.get_pipe()->request_autocontrast(compute_desc_.current_window);
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
					compute_desc_.set_contrast_min(compute_desc_.current_window, value);
					pipe_refresh();
				}
			}
		}

		void MainWindow::set_contrast_max(const double value)
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.contrast_enabled)
				{
					compute_desc_.set_contrast_max(compute_desc_.current_window, value);
					pipe_refresh();
				}
			}
		}

		void MainWindow::invert_contrast(bool value)
		{
			if (!is_direct_mode())
			{
				if (compute_desc_.contrast_enabled)
				{
					compute_desc_.contrast_invert = value;
					pipe_refresh();
				}
			}
		}

		void MainWindow::set_log_scale(const bool value)
		{
			if (!is_direct_mode())
			{
				compute_desc_.set_log_scale_slice_enabled(compute_desc_.current_window, value);
				if (compute_desc_.contrast_enabled)
				{
					set_contrast_min(ui.ContrastMinDoubleSpinBox->value());
					set_contrast_max(ui.ContrastMaxDoubleSpinBox->value());
				}
				pipe_refresh();
				set_auto_contrast();
				notify();
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
				{
					mainDisplay->getOverlayManager().disable_all(Signal);
					mainDisplay->getOverlayManager().disable_all(Noise);
				}
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
			ui.ConvoCheckBox->setChecked(false);
			holovibes_.reset_convolution_matrix();

			try
			{
				std::ifstream file(path);

				strStream << file.rdbuf();
				file.close();
				str = strStream.str();
				boost::split(v_str, str, boost::is_any_of(";"));
				if (v_str.size() != 2)
				{
					display_error("Couldn't load file : too many or not enough separators \";\"\n");
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

				uint matrix_width = std::stoi(matrix_size[0]);
				uint matrix_height = std::stoi(matrix_size[1]);
				uint matrix_z = std::stoi(matrix_size[2]);
				boost::trim(v_str[1]);
				boost::split(matrix, v_str[1], boost::is_any_of(delims), boost::token_compress_on);
				if (matrix_width * matrix_height * matrix_z != matrix.size())
				{
					holovibes_.reset_convolution_matrix();
					display_error("Couldn't load file : the dimension and the number of elements in the matrix\n");
				}

				//on plonge le kernel dans un carre de taille nx*ny tout en gardant le profondeur z
				//TODO a paralleliser
				uint h = 0;
				uint c = 0;
				uint nx = ui.ImportWidthSpinBox->value();
				uint ny = ui.ImportHeightSpinBox->value();
				uint size = nx * ny;

				std::vector<float> convo_matrix(size, 0.0f);

				const  uint minw = (nx / 2) - (matrix_width / 2);
				const  uint maxw = (nx / 2) + (matrix_width / 2);
				const  uint minh = (ny / 2) - (matrix_height / 2);
				const  uint maxh = (ny / 2) + (matrix_height / 2);

				for (size_t i = minw; i < maxw; i++)
				{
					for (size_t j = minh; j < maxh; j++)
					{
						convo_matrix[i * nx + j] = std::stof(matrix[c]);
						c++;
					}
				}

				//on met les largeurs et hauteurs a la taille de nx et de ny
				compute_desc_.convo_matrix_width = nx;
				compute_desc_.convo_matrix_height = ny;
				compute_desc_.convo_matrix_z = matrix_z;
				compute_desc_.convo_matrix = convo_matrix;

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
				tr("Record output file"), "C://", tr("Holo files (*.holo);; All files (*)"));

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
			std::string mode = (is_direct_mode() || compute_desc_.record_raw) ? "D" : "H";

			std::string sub_str = "_" + slice
				+ "_" + mode
				+ "_" + std::to_string(fd.width)
				+ "_" + std::to_string(fd.height);
			int depth = fd.depth;
			if (depth == 6)
				depth = 3;
			sub_str += "_" + std::to_string(depth << 3) + "bit"
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

		void MainWindow::set_raw_recording(bool value)
		{
			compute_desc_.record_raw = value;
		}

		void MainWindow::set_synchronized_record(bool value)
		{
			compute_desc_.synchronized_record = value;
		}

		void MainWindow::normalize(bool value)
		{
			compute_desc_.normalize_enabled = value;
		}

		void MainWindow::display_cross(bool value)
		{
			ui.ReticleScaleDoubleSpinBox->setEnabled(value);
			if (value)
			{
				mainDisplay->getOverlayManager().create_overlay<Reticle>();
			}
			else
			{
				mainDisplay->getOverlayManager().disable_all(Reticle);
			}
			notify();
		}

		void MainWindow::reticle_scale(double value)
		{
			if (0 > value || value > 1)
				return;
			compute_desc_.reticle_scale = value;
		}

		void MainWindow::start_recording()
		{
			record_thread_->start();
			ui.ImageOutputStopPushButton->setDisabled(false);
			auto reader = dynamic_cast<ThreadReader *>(holovibes_.get_tcapture().get());
			if (reader)
				disconnect(reader, SIGNAL(at_begin()), this, SLOT(start_recording()));
		}

		void MainWindow::set_record()
		{
			QSpinBox*  nb_of_frames_spinbox = ui.NumberOfFramesSpinBox;
			QLineEdit* path_line_edit = ui.ImageOutputPathLineEdit;

			int nb_of_frames = nb_of_frames_spinbox->value();
			std::string path = path_line_edit->text().toUtf8();
			if (path == "")
				return display_error("No output file");

			try
			{
				Queue *queue;
				if (compute_desc_.record_raw)
				{
					queue = holovibes_.get_pipe()->get_raw_queue().get();
					queue->set_display(true);
				}
				else
					queue = holovibes_.get_current_window_output_queue().get();

				if (queue)
				{
					// path = set_record_filename_properties(queue->get_frame_desc(), path);
					if (path.substr(path.size() - 5, 5) != ".holo")
						path += ".holo";
					record_thread_.reset(new ThreadRecorder(*queue, path, nb_of_frames, holo_file_get_json_settings(), this));

					connect(record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_image_record()));
					if (compute_desc_.synchronized_record)
					{
						auto reader = dynamic_cast<ThreadReader *>(holovibes_.get_tcapture().get());
						if (reader)
							connect(reader, SIGNAL(at_begin()), this, SLOT(start_recording()));
					}
					else
					{
						record_thread_->start();
						ui.ImageOutputStopPushButton->setDisabled(false);
					}

					ui.RawRecordingCheckBox->setDisabled(true);
					ui.SynchronizedRecordCheckBox->setDisabled(true);
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

			ui.ImageOutputStopPushButton->setDisabled(true);

			if (compute_desc_.record_raw && !compute_desc_.raw_view)
			{
				holovibes_.get_pipe()->get_raw_queue()->set_display(false);
				gui::InfoManager::get_manager()->remove_info("RawOutputQueue");
			}

			ui.RawRecordingCheckBox->setDisabled(false);

			ui.SynchronizedRecordCheckBox->setDisabled(false);

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
					q = holovibes_.get_output_queue().get();
				else if (compute_desc_.current_window == WindowKind::XZview)
					q = holovibes_.get_pipe()->get_stft_slice_queue(0).get();
				else
					q = holovibes_.get_pipe()->get_stft_slice_queue(1).get();
				// Only loading the dll at runtime
				gpib_interface_ = gpib::GpibDLL::load_gpib("gpib.dll", input_path);

				formatted_path = format_batch_output(path, file_index_);
				formatted_path = set_record_filename_properties(q->get_frame_desc(), formatted_path);

				//is_enabled_camera_ = false;

				if (gpib_interface_->execute_next_block()) // More blocks to come, use batch_next_block method.
				{
					if (is_batch_img_)
					{
						record_thread_.reset(new ThreadRecorder(*q, formatted_path, frame_nb, holo_file_get_json_settings(), this));
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
						record_thread_.reset(new ThreadRecorder(*q, formatted_path, frame_nb, holo_file_get_json_settings(), this));
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
				q = holovibes_.get_output_queue().get();
			else if (compute_desc_.current_window == WindowKind::XZview)
				q = holovibes_.get_pipe()->get_stft_slice_queue(0).get();
			else
				q = holovibes_.get_pipe()->get_stft_slice_queue(1).get();

			std::string output_filename = format_batch_output(path, file_index_);
			output_filename = set_record_filename_properties(q->get_frame_desc(), output_filename);
			const uint frame_nb = frame_nb_spin_box->value();
			if (is_batch_img_)
			{
				try
				{
					if (gpib_interface_->execute_next_block())
					{
						record_thread_.reset(new ThreadRecorder(*q, output_filename, frame_nb, holo_file_get_json_settings(), this));
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

			if (filename != "" && filename != tmp_path)
			{
				import_line_edit->clear();
				import_line_edit->insert(filename);
				tmp_path = filename;

				auto holo_file = HoloFile::new_instance(filename.toStdString());
				compute_desc_.is_holo_file = holo_file;
				holo_file_update_ui();

				if (!holo_file)
				{
					title_detect();
				}
			}
		}

		void MainWindow::import_file_stop(void)
		{
			close_windows();
			close_critical_compute();
			camera_none();
			remove_infos();
			compute_desc_.compute_mode = Computation::Stop;
			notify();
		}

		void MainWindow::import_file()
		{
			holo_file_update_cd();
			import_file_stop();
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

			width = static_cast<ushort>(width_spinbox->value());
			height = static_cast<ushort>(height_spinbox->value());
			//modify the third parameter to change the width and the height of the Holowindow
			get_good_size(width, height, 512);

			//the convolution is disabled to avoid problem with iamge size
			ui.ConvoCheckBox->setChecked(false);
			set_convolution_mode(false);

			ui.ToHoloFilePushButton->setEnabled(!HoloFile::get_instance());
			ui.UpdateHoloPushButton->setEnabled(HoloFile::get_instance());

			compute_desc_.stft_steps = std::ceil(static_cast<float>(fps_spinbox->value()) / 20.0f);
			compute_desc_.pixel_size = pixel_size_spinbox->value();
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
				static_cast<unsigned int>(depth_multi),
				(big_endian_checkbox->currentText() == QString("Big Endian") ?
					Endianness::BigEndian : Endianness::LittleEndian) };
			is_enabled_camera_ = false;
			try
			{
				auto file_end = std::filesystem::file_size(file_src)
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

			holo_file_update_cd();

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

		void MainWindow::to_holo_file()
		{
			unsigned width = ui.ImportWidthSpinBox->value();
			unsigned height = ui.ImportHeightSpinBox->value();
			unsigned pixel_bits = std::pow(2, ui.ImportDepthComboBox->currentIndex() + 3);
			auto header = HoloFile::create_header(pixel_bits, width, height);
			HoloFile::create(header, holo_file_get_json_settings().dump(), ui.ImportPathLineEdit->text().toStdString());
		}

		void MainWindow::holo_file_update_ui()
		{
			auto holo_file = HoloFile::get_instance();

			ui.ToHoloFilePushButton->setEnabled(!holo_file && compute_desc_.compute_mode != Computation::Stop);
			ui.UpdateHoloPushButton->setEnabled(holo_file && compute_desc_.compute_mode != Computation::Stop);

			if (!holo_file)
				return;

			const HoloFile::Header& header = holo_file.get_header();
			const json& json_settings = holo_file.get_meta_data();
			ui.ImportWidthSpinBox->setValue(header.img_width);
			ui.ImportHeightSpinBox->setValue(header.img_height);
			ui.ImportDepthComboBox->setCurrentIndex(log2(header.pixel_bits) - 3);
			ui.ImportEndiannessComboBox->setCurrentIndex(json_settings.value("endianess", 0));
		}

		void MainWindow::holo_file_update_cd()
		{
			auto holo_file = HoloFile::get_instance();

			if (!holo_file)
				return;

			const json& json_settings = holo_file.get_meta_data();
			compute_desc_.algorithm = static_cast<Algorithm>(json_settings.value("algorithm", 0));
			compute_desc_.nSize = json_settings.value("#img", 1);
			compute_desc_.pindex = json_settings.value("p", 0);
			compute_desc_.lambda = json_settings.value("lambda", 0.0f);
			compute_desc_.zdistance = json_settings.value("z", 0.0f);
			compute_desc_.log_scale_slice_xy_enabled = json_settings.value("log_scale", false);
			compute_desc_.contrast_min_slice_xy = json_settings.value("contrast_min", 0.0f);
			compute_desc_.contrast_max_slice_xy = json_settings.value("contrast_max", 0.0f);
		}

		json MainWindow::holo_file_get_json_settings()
		{
			try
			{
				json json_settings;
				auto& output_queue = holovibes_.get_output_queue();
				if (output_queue != nullptr)
				{
					json_settings = HoloFile::get_json_settings(compute_desc_, output_queue->get_frame_desc());
				}
				else
				{
					// This code shouldn't run but it's here to avoid a segfault in case something weird happens
					json_settings = HoloFile::get_json_settings(compute_desc_);
					json_settings.emplace("img_width", ui.ImportWidthSpinBox->value());
					json_settings.emplace("img_height", ui.ImportHeightSpinBox->value());
					json_settings.emplace("pixel_bits", std::pow(2, ui.ImportDepthComboBox->currentIndex() + 3));
					json_settings.emplace("endianess", ui.ImportEndiannessComboBox->currentIndex());
				}
				return json_settings;
			}
			catch (const std::exception& e)
			{
				LOG_ERROR(e.what());
				return json();
			}
		}

		void MainWindow::holo_file_update()
		{
			HoloFile::get_instance().update(holo_file_get_json_settings().dump());
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
			auto lambda = [this, n]() {
				ui.FileReaderProgressBar->setValue(n);
			};
			synchronize_thread(lambda);
		}
#pragma endregion
	}
}
