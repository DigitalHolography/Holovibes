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
#include <algorithm>
#include <list>

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
#include "config.hh"
#include "cine_file.hh"

#define MIN_IMG_NB_TIME_FILTER_CUTS 8

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
			window_max_size(768),
			time_filter_cuts_window_max_size(512),
			auxiliary_window_max_size(512),
			sliceXZ(nullptr),
			sliceYZ(nullptr),
			displayAngle(0.f),
			xzAngle(0.f),
			yzAngle(0.f),
			displayFlip(0),
			xzFlip(0),
			yzFlip(0),
			is_enabled_camera_(false),
			is_batch_img_(true),
			is_batch_interrupted_(false),
			z_step_(0.005f),
			record_frame_step_(1024),
			kCamera(CameraKind::NONE),
			last_img_type_("Magnitude"),
			plot_window_(nullptr),
			record_thread_(nullptr),
			CSV_record_thread_(nullptr),
			auto_scale_point_threshold_(100),
			file_index_(1),
			theme_index_(0),
			import_type_(ImportType::None),
			cd_(holovibes_.get_cd())
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

			QComboBox *window_cbox = ui.WindowSelectionComboBox;
			connect(window_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(change_window()));

			// Display default values
			cd_.compute_mode = Computation::Raw;
			notify();
			cd_.compute_mode = Computation::Stop;
			notify();
			setFocusPolicy(Qt::StrongFocus);

			// spinBox allow ',' and '.' as decimal point
			spinBoxDecimalPointReplacement(ui.WaveLengthDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.ZDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.ZStepDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.ContrastMaxDoubleSpinBox);
			spinBoxDecimalPointReplacement(ui.ContrastMinDoubleSpinBox);

			ui.FileReaderProgressBar->hide();
			ui.ExportProgressBar;

			// Fill the quick kernel combo box with files from convolution_kernels directory
			std::filesystem::path convo_matrix_path(get_exe_dir());
			convo_matrix_path = convo_matrix_path / "convolution_kernels";
			if (std::filesystem::exists(convo_matrix_path))
			{
				QVector<QString> files;
				for (const auto& file : std::filesystem::directory_iterator(convo_matrix_path))
				{
					files.push_back(QString(file.path().filename().string().c_str()));
				}
				std::sort(files.begin(), files.end(), [&](const auto& a, const auto& b) { return a < b; });
				ui.KernelQuickSelectComboBox->addItems(QStringList::fromVector(files));
			}
		}

		MainWindow::~MainWindow()
		{
			delete z_up_shortcut_;
			delete z_down_shortcut_;
			delete p_left_shortcut_;
			delete p_right_shortcut_;

			close_windows();
			close_critical_compute();
			camera_none();
			remove_infos();

			holovibes_.dispose_compute();
			if (!is_raw_mode())
				holovibes_.dispose_capture();
			InfoManager::get_manager()->stop_display();
		}


#pragma endregion
		/* ------------ */
#pragma region Notify
		void MainWindow::synchronize_thread(std::function<void()> f)
		{
			// We can't update gui values from a different thread
			// so we pass it to the right one using a signal
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
			const bool is_raw = is_raw_mode();

			// Tabs
			if (cd_.compute_mode == Computation::Stop)
			{
				ui.ImageRenderingGroupBox->setEnabled(false);
				ui.ViewGroupBox->setEnabled(false);
				ui.ExportGroupBox->setEnabled(false);
				ui.ImportGroupBox->setEnabled(true);
				ui.InfoGroupBox->setEnabled(true);
				return;
			}
			else if (cd_.compute_mode == Computation::Raw && is_enabled_camera_)
			{
				ui.ImageRenderingGroupBox->setEnabled(true);
				ui.ExportGroupBox->setEnabled(true);
			}
			else if (cd_.compute_mode == Computation::Hologram && is_enabled_camera_)
			{
				ui.ImageRenderingGroupBox->setEnabled(true);
				ui.ViewGroupBox->setEnabled(true);
				ui.ExportGroupBox->setEnabled(true);
			}

			// Record
			ui.RawRecordingCheckBox->setEnabled(!is_raw);
			ui.RawRecordingCheckBox->setChecked(!is_raw && cd_.record_raw);
			ui.SynchronizedRecordCheckBox->setEnabled(import_type_ == File);
			ui.SynchronizedRecordCheckBox->setChecked(import_type_ == File
				&& cd_.synchronized_record);

			// Raw view
			ui.RawDisplayingCheckBox->setEnabled(!is_raw && !cd_.record_raw);
			ui.RawDisplayingCheckBox->setChecked(!is_raw && cd_.raw_view);

			// Chart
			ui.ChartGroupBox->setChecked(!is_raw && cd_.chart_enabled);
			if (mainDisplay
				&& mainDisplay->getOverlayManager().is_signal_zone_set()
				&& mainDisplay->getOverlayManager().is_noise_zone_set())
				set_chart_graphic();

			QPushButton* signalBtn = ui.ChartSignalPushButton;
			signalBtn->setStyleSheet((signalBtn->isEnabled() &&
				mainDisplay && mainDisplay->getKindOfOverlay() == KindOfOverlay::Signal) ? "QPushButton {color: #8E66D9;}" : "");

			QPushButton* noiseBtn = ui.ChartNoisePushButton;
			noiseBtn->setStyleSheet((noiseBtn->isEnabled() &&
				mainDisplay && mainDisplay->getKindOfOverlay() == KindOfOverlay::Noise) ? "QPushButton {color: #00A4AB;}" : "");

			// Displaying mode
			ui.ViewModeComboBox->setCurrentIndex(cd_.img_type);

			ui.PhaseUnwrap2DCheckBox->
				setEnabled(cd_.img_type == ImgType::PhaseIncrease ||
					cd_.img_type == ImgType::Argument);

			// Time filter cuts
			ui.TimeFilterCutsCheckBox->setChecked(!is_raw && cd_.time_filter_cuts_enabled);

			// Contrast
			ui.ContrastCheckBox->setChecked(!is_raw && cd_.contrast_enabled);
			ui.ContrastCheckBox->setEnabled(true);
			ui.AutoRefreshContrastCheckBox->setChecked(cd_.contrast_auto_refresh);

			// Contrast SpinBox:
			ui.ContrastMinDoubleSpinBox->setEnabled(!cd_.contrast_auto_refresh);
			ui.ContrastMinDoubleSpinBox->setValue(cd_.get_contrast_min(cd_.current_window));
			ui.ContrastMaxDoubleSpinBox->setEnabled(!cd_.contrast_auto_refresh);
			ui.ContrastMaxDoubleSpinBox->setValue(cd_.get_contrast_max(cd_.current_window));

			// FFT shift
			ui.FFTShiftCheckBox->setChecked(cd_.fft_shift_enabled);
			ui.FFTShiftCheckBox->setEnabled(true);

			// Window selection
			QComboBox *window_selection = ui.WindowSelectionComboBox;
			window_selection->setEnabled(cd_.time_filter_cuts_enabled);
			window_selection->setCurrentIndex(window_selection->isEnabled() ? cd_.current_window : 0);

			ui.LogScaleCheckBox->setEnabled(true);
			ui.LogScaleCheckBox->setChecked(!is_raw && cd_.get_img_log_scale_slice_enabled(cd_.current_window));
			ui.ImgAccuCheckBox->setEnabled(true);
			ui.ImgAccuCheckBox->setChecked(!is_raw && cd_.get_img_acc_slice_enabled(cd_.current_window));
			ui.ImgAccuSpinBox->setValue(cd_.get_img_acc_slice_level(cd_.current_window));
			if (cd_.current_window == WindowKind::XYview)
			{
				ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(displayAngle))).c_str());
				ui.FlipPushButton->setText(("Flip " + std::to_string(displayFlip)).c_str());
			}
			else if (cd_.current_window == WindowKind::XZview)
			{
				ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(xzAngle))).c_str());
				ui.FlipPushButton->setText(("Flip " + std::to_string(xzFlip)).c_str());
			}
			else if (cd_.current_window == WindowKind::YZview)
			{
				ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(yzAngle))).c_str());
				ui.FlipPushButton->setText(("Flip " + std::to_string(yzFlip)).c_str());
			}

			// p accu
			ui.PAccuCheckBox->setEnabled(cd_.img_type != PhaseIncrease);
			ui.PAccuCheckBox->setChecked(cd_.p_accu_enabled);
			ui.PAccSpinBox->setMaximum(cd_.time_filter_size - 1);
			if (cd_.p_acc_level > cd_.time_filter_size - 1)
				cd_.p_acc_level = cd_.time_filter_size - 1;
			ui.PAccSpinBox->setValue(cd_.p_acc_level);
			ui.PAccSpinBox->setEnabled(cd_.img_type != PhaseIncrease);
			if (cd_.p_accu_enabled)
			{
				ui.PSpinBox->setMaximum(cd_.time_filter_size - cd_.p_acc_level - 1);
				if (cd_.pindex > cd_.time_filter_size - cd_.p_acc_level - 1)
					cd_.pindex = cd_.time_filter_size - cd_.p_acc_level - 1;
				ui.PSpinBox->setValue(cd_.pindex);
				ui.PAccSpinBox->setMaximum(cd_.time_filter_size - cd_.pindex - 1);
			}
			else
			{
				ui.PSpinBox->setMaximum(cd_.time_filter_size - 1);
				if (cd_.pindex > cd_.time_filter_size - 1)
					cd_.pindex = cd_.time_filter_size - 1;
				ui.PSpinBox->setValue(cd_.pindex);
			}

			// XY accu
			ui.XAccuCheckBox->setChecked(cd_.x_accu_enabled);
			ui.XAccSpinBox->setValue(cd_.x_acc_level);
			ui.YAccuCheckBox->setChecked(cd_.y_accu_enabled);
			ui.YAccSpinBox->setValue(cd_.y_acc_level);

			// STFT
			ui.STFTStepsSpinBox->setEnabled(!is_raw);

			const uint input_queue_capacity = global::global_config.input_queue_max_size;
			if (cd_.time_filter_stride > input_queue_capacity)
				cd_.time_filter_stride = input_queue_capacity;

			ui.STFTStepsSpinBox->setValue(cd_.time_filter_stride);
			ui.STFTStepsSpinBox->setSingleStep(cd_.batch_size);
			ui.STFTStepsSpinBox->setMinimum(cd_.batch_size);
			ui.STFTStepsSpinBox->setMaximum(input_queue_capacity);

			// Batch
			ui.BatchSizeSpinBox->setEnabled(!is_raw);

			if (cd_.batch_size > input_queue_capacity)
				cd_.batch_size = input_queue_capacity;

			ui.BatchSizeSpinBox->setValue(cd_.batch_size);
			ui.BatchSizeSpinBox->setMaximum(input_queue_capacity);

			// Image rendering
			ui.AlgorithmComboBox->setEnabled(!is_raw);
			ui.AlgorithmComboBox->setCurrentIndex(cd_.algorithm);
			ui.TimeAlgorithmComboBox->setEnabled(!is_raw);
			ui.TimeAlgorithmComboBox->setCurrentIndex(cd_.time_filter);

			// Changing time_filter_size with time filter cuts is supported by the pipe, but some modifications have to be done in SliceWindow, OpenGl buffers.
			ui.timeFilterSizeSpinBox->setEnabled(!is_raw && !cd_.time_filter_cuts_enabled);
			ui.timeFilterSizeSpinBox->setValue(cd_.time_filter_size);
			ui.TimeFilterCutsCheckBox->setEnabled(ui.timeFilterSizeSpinBox->value() >= MIN_IMG_NB_TIME_FILTER_CUTS);

			ui.WaveLengthDoubleSpinBox->setEnabled(!is_raw);
			ui.WaveLengthDoubleSpinBox->setValue(cd_.lambda * 1.0e9f);
			ui.ZDoubleSpinBox->setEnabled(!is_raw);
			ui.ZDoubleSpinBox->setValue(cd_.zdistance);
			ui.ZStepDoubleSpinBox->setEnabled(!is_raw);
			ui.BoundaryLineEdit->setText(QString::number(holovibes_.get_boundary()));

			// Filter2d
			QPushButton *filter_button = ui.Filter2DPushButton;
			filter_button->setEnabled(!is_raw && !cd_.filter_2d_enabled);
			filter_button->setStyleSheet((!is_raw && cd_.filter_2d_enabled) ? "QPushButton {color: #009FFF;}" : "");
			ui.CancelFilter2DPushButton->setEnabled(!is_raw && cd_.filter_2d_enabled);

			// Composite
			const int time_filter_size_max = cd_.time_filter_size - 1;
			ui.PRedSpinBox_Composite->setMaximum(time_filter_size_max);
			ui.PBlueSpinBox_Composite->setMaximum(time_filter_size_max);
			ui.SpinBox_hue_freq_min->setMaximum(time_filter_size_max);
			ui.SpinBox_hue_freq_max->setMaximum(time_filter_size_max);
			ui.SpinBox_saturation_freq_min->setMaximum(time_filter_size_max);
			ui.SpinBox_saturation_freq_max->setMaximum(time_filter_size_max);
			ui.SpinBox_value_freq_min->setMaximum(time_filter_size_max);
			ui.SpinBox_value_freq_max->setMaximum(time_filter_size_max);

			ui.RenormalizationCheckBox->setChecked(cd_.composite_auto_weights_);

			QSpinBoxQuietSetValue(ui.PRedSpinBox_Composite, cd_.composite_p_red);
			QSpinBoxQuietSetValue(ui.PBlueSpinBox_Composite, cd_.composite_p_blue);
			QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_R, cd_.weight_r);
			QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_G, cd_.weight_g);
			QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_B, cd_.weight_b);
			actualize_frequency_channel_v();

			QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_min, cd_.composite_p_min_h);
			QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_max, cd_.composite_p_max_h);
			QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_min, (int)(cd_.slider_h_threshold_min * 1000));
			slide_update_threshold_h_min();
			QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_max, (int)(cd_.slider_h_threshold_max * 1000));
			slide_update_threshold_h_max();

			QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_min, cd_.composite_p_min_s);
			QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_max, cd_.composite_p_max_s);
			QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_min, (int)(cd_.slider_s_threshold_min * 1000));
			slide_update_threshold_s_min();
			QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_max, (int)(cd_.slider_s_threshold_max * 1000));
			slide_update_threshold_s_max();

			QSpinBoxQuietSetValue(ui.SpinBox_value_freq_min, cd_.composite_p_min_v);
			QSpinBoxQuietSetValue(ui.SpinBox_value_freq_max, cd_.composite_p_max_v);
			QSliderQuietSetValue(ui.horizontalSlider_value_threshold_min, (int)(cd_.slider_v_threshold_min * 1000));
			slide_update_threshold_v_min();
			QSliderQuietSetValue(ui.horizontalSlider_value_threshold_max, (int)(cd_.slider_v_threshold_max * 1000));
			slide_update_threshold_v_max();


			ui.CompositeGroupBox->setHidden(is_raw_mode() || (cd_.img_type != ImgType::Composite));

			bool rgbMode = ui.radioButton_rgb->isChecked();
			ui.groupBox->setHidden(!rgbMode);
			ui.groupBox_5->setHidden(!rgbMode && !ui.RenormalizationCheckBox->isChecked());
			ui.groupBox_hue->setHidden(rgbMode);
			ui.groupBox_saturation->setHidden(rgbMode);
			ui.groupBox_value->setHidden(rgbMode);

			// Reticle
			if (ui.DisplayReticleCheckBox->isChecked()
				&& mainDisplay
				&& mainDisplay->getOverlayManager().getKind() != KindOfOverlay::Reticle)
			{
				ui.DisplayReticleCheckBox->setChecked(false);
				display_cross(false);
			}

			// Lens View
			ui.LensViewCheckBox->setChecked(cd_.gpu_lens_display_enabled);

			// Renormalize
			ui.RenormalizeCheckBox->setChecked(cd_.renorm_enabled);
			ui.RenormalizeSpinBox->setValue(cd_.renorm_constant);

			// Convolution
			ui.ConvoCheckBox->setChecked(cd_.convolution_enabled);
			ui.DivideConvoCheckBox->setChecked(cd_.convolution_enabled && cd_.divide_convolution_enabled);
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
						cd_.pindex = 0;
						cd_.time_filter_size = 1;
						if (cd_.convolution_enabled)
						{
							cd_.convolution_enabled = false;
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
						cd_.img_acc_slice_xy_enabled = false;
						cd_.img_acc_slice_xy_level = 1;
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

				"Anthony Strazzella\n"
				"Ilan Guenet\n"
				"Nicolas Blin\n"
				"Quentin Kaci\n"
				"Theo Lepage\n"

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
			QDesktopServices::openUrl(QUrl("https://ftp.espci.fr/incoming/Atlan/holovibes/manual/"));
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
			import_stop();
			try
			{
				load_ini(GLOBAL_INI_PATH);
			}
			catch (std::exception& e)
			{
				std::cout << e.what() << std::endl;
			}
			if (import_type_ == ImportType::File)
			{
				import_start();
			}
			else if (import_type_ == ImportType::Camera)
			{
				change_camera(kCamera);
			}
			notify();
		}

		void MainWindow::reset_input()
		{
			import_stop();

			if (import_type_ == ImportType::File)
			{
				set_computation_mode();
				init_holovibes_import_mode();
			}
			else if (import_type_ == ImportType::Camera)
				change_camera(kCamera);

			notify();
		}

		void MainWindow::load_ini(const std::string& path)
		{
			boost::property_tree::ptree ptree;
			GroupBox *image_rendering_group_box = ui.ImageRenderingGroupBox;
			GroupBox *view_group_box = ui.ViewGroupBox;
			GroupBox *record_group_box = ui.ExportGroupBox;
			GroupBox *import_group_box = ui.ImportGroupBox;
			GroupBox *info_group_box = ui.InfoGroupBox;

			QAction	*image_rendering_action = ui.actionImage_rendering;
			QAction	*view_action = ui.actionView;
			QAction	*export_action = ui.actionExport;
			QAction	*import_action = ui.actionImport;
			QAction	*info_action = ui.actionInfo;

			boost::property_tree::ini_parser::read_ini(path, ptree);

			if (!ptree.empty())
			{
				Config& config = global::global_config;
				// Config
				config.file_buffer_size = ptree.get<int>("config.file_buffer_size", config.file_buffer_size);
				config.input_queue_max_size = ptree.get<int>("config.input_buffer_size", config.input_queue_max_size);
				config.output_queue_max_size = ptree.get<int>("config.output_buffer_size", config.output_queue_max_size);
				config.time_filter_cuts_output_buffer_size = ptree.get<int>("config.time_filter_cuts_output_buffer_size", config.time_filter_cuts_output_buffer_size);
				config.frame_timeout = ptree.get<int>("config.frame_timeout", config.frame_timeout);
				config.flush_on_refresh = ptree.get<int>("config.flush_on_refresh", config.flush_on_refresh);

				cd_.img_acc_slice_xy_level = ptree.get<uint>("config.accumulation_buffer_size", cd_.img_acc_slice_xy_level);
				cd_.display_rate = ptree.get<float>("config.display_rate", cd_.display_rate);

				// Image rendering
				image_rendering_action->setChecked(!ptree.get<bool>("image_rendering.hidden", image_rendering_group_box->isHidden()));

				const ushort p_time_filter_size = ptree.get<ushort>("image_rendering.time_filter_size", cd_.time_filter_size);
				if (p_time_filter_size < 1)
					cd_.time_filter_size = 1;
				else
					cd_.time_filter_size = p_time_filter_size;
				const ushort p_index = ptree.get<ushort>("image_rendering.p_index", cd_.pindex);
				if (p_index >= 0 && p_index < cd_.time_filter_size)
					cd_.pindex = p_index;

				cd_.lambda = ptree.get<float>("image_rendering.lambda", cd_.lambda);

				cd_.zdistance = ptree.get<float>("image_rendering.z_distance", cd_.zdistance);

				const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
				if (z_step > 0.0f)
					set_z_step(z_step);

				cd_.algorithm = static_cast<Algorithm>(ptree.get<int>("image_rendering.algorithm", cd_.algorithm));

				cd_.raw_bitshift = ptree.get<ushort>("image_rendering.raw_bitshift", cd_.raw_bitshift);

				cd_.time_filter_stride = ptree.get<int>("image_rendering.time_filter_stride", cd_.time_filter_stride);

				// View
				view_action->setChecked(!ptree.get<bool>("view.hidden", view_group_box->isHidden()));

				cd_.img_type.exchange(static_cast<ImgType>(
					ptree.get<int>("view.view_mode", cd_.img_type)));
				last_img_type_ = cd_.img_type == ImgType::Composite ? "Composite image" : last_img_type_;

				cd_.log_scale_slice_xy_enabled = ptree.get<bool>("view.log_scale_enabled", cd_.log_scale_slice_xy_enabled);
				cd_.log_scale_slice_xz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_xz", cd_.log_scale_slice_xz_enabled);
				cd_.log_scale_slice_yz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_yz", cd_.log_scale_slice_yz_enabled);

				cd_.fft_shift_enabled = ptree.get<bool>("view.fft_shift_enabled", cd_.fft_shift_enabled);

				cd_.contrast_enabled = ptree.get<bool>("view.contrast_enabled", cd_.contrast_enabled);
				cd_.contrast_threshold_low_percentile = ptree.get<float>("view.contrast_threshold_low_percentile", cd_.contrast_threshold_low_percentile);
				cd_.contrast_threshold_high_percentile = ptree.get<float>("view.contrast_threshold_high_percentile", cd_.contrast_threshold_high_percentile);

				cd_.contrast_min_slice_xy = ptree.get<float>("view.contrast_min", cd_.contrast_min_slice_xy);
				cd_.contrast_max_slice_xy = ptree.get<float>("view.contrast_max", cd_.contrast_max_slice_xy);
				cd_.cuts_contrast_p_offset = ptree.get<ushort>("view.cuts_contrast_p_offset", cd_.cuts_contrast_p_offset);
				if (cd_.cuts_contrast_p_offset < 0)
					cd_.cuts_contrast_p_offset = 0;
				else if (cd_.cuts_contrast_p_offset > cd_.time_filter_size - 1)
					cd_.cuts_contrast_p_offset = cd_.time_filter_size - 1;

				cd_.img_acc_slice_xy_enabled = ptree.get<bool>("view.accumulation_enabled", cd_.img_acc_slice_xy_enabled);

				displayAngle = ptree.get("view.mainWindow_rotate", displayAngle);
				xzAngle = ptree.get<float>("view.xCut_rotate", xzAngle);
				yzAngle = ptree.get<float>("view.yCut_rotate", yzAngle);
				displayFlip = ptree.get("view.mainWindow_flip", displayFlip);
				xzFlip = ptree.get("view.xCut_flip", xzFlip);
				yzFlip = ptree.get("view.yCut_flip", yzFlip);
				cd_.reticle_scale = ptree.get("view.reticle_scale", 0.5f);

				// Chart
				auto_scale_point_threshold_ = ptree.get<size_t>("chart.auto_scale_point_threshold", auto_scale_point_threshold_);

				// Record
				export_action->setChecked(!ptree.get<bool>("record.hidden", record_group_box->isHidden()));

				const uint record_frame_step = ptree.get<uint>("record.record_frame_step", record_frame_step_);
				set_record_frame_step(record_frame_step);

				// Import
				import_action->setChecked(!ptree.get<bool>("import.hidden", import_group_box->isHidden()));
				cd_.pixel_size = ptree.get<float>("import.pixel_size", cd_.pixel_size);
				ui.ImportInputFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));

				// Info
				info_action->setChecked(!ptree.get<bool>("info.hidden", info_group_box->isHidden()));
				theme_index_ = ptree.get<int>("info.theme_type", theme_index_);

				// Reset button
				config.set_cuda_device = ptree.get<bool>("reset.set_cuda_device", config.set_cuda_device);
				config.auto_device_number = ptree.get<bool>("reset.auto_device_number", config.auto_device_number);
				config.device_number = ptree.get<int>("reset.device_number", config.device_number);

				// Composite
				cd_.composite_p_red = ptree.get<ushort>("composite.p_red", 1);
				cd_.composite_p_blue = ptree.get<ushort>("composite.p_blue", 1);
				cd_.weight_r = ptree.get<float>("composite.weight_r", 1);
				cd_.weight_g = ptree.get<float>("composite.weight_g", 1);
				cd_.weight_b = ptree.get<float>("composite.weight_b", 1);

				cd_.composite_p_min_h = ptree.get<ushort>("composite.p_min_h", 1);
				cd_.composite_p_max_h = ptree.get<ushort>("composite.p_max_h", 1);
				cd_.slider_h_threshold_min = ptree.get<float>("composite.slider_h_threshold_min", 0);
				cd_.slider_h_threshold_max = ptree.get<float>("composite.slider_h_threshold_max", 1.0f);
				cd_.composite_low_h_threshold = ptree.get<float>("composite.low_h_threshold", 0.2f);
				cd_.composite_high_h_threshold = ptree.get<float>("composite.high_h_threshold", 99.8f);

				cd_.composite_p_activated_s = ptree.get<bool>("composite.p_activated_s", false);
				cd_.composite_p_min_s = ptree.get<ushort>("composite.p_min_s", 1);
				cd_.composite_p_max_s = ptree.get<ushort>("composite.p_max_s", 1);
				cd_.slider_s_threshold_min = ptree.get<float>("composite.slider_s_threshold_min", 0);
				cd_.slider_s_threshold_max = ptree.get<float>("composite.slider_s_threshold_max", 1.0f);
				cd_.composite_low_s_threshold = ptree.get<float>("composite.low_s_threshold", 0.2f);
				cd_.composite_high_s_threshold = ptree.get<float>("composite.high_s_threshold", 99.8f);

				cd_.composite_p_activated_v = ptree.get<bool>("composite.p_activated_v", false);
				cd_.composite_p_min_v = ptree.get<ushort>("composite.p_min_v", 1);
				cd_.composite_p_max_v = ptree.get<ushort>("composite.p_max_v", 1);
				cd_.slider_v_threshold_min = ptree.get<float>("composite.slider_v_threshold_min", 0);
				cd_.slider_v_threshold_max = ptree.get<float>("composite.slider_v_threshold_max", 1.0f);
				cd_.composite_low_v_threshold = ptree.get<float>("composite.low_v_threshold", 0.2f);
				cd_.composite_high_v_threshold = ptree.get<float>("composite.high_v_threshold", 99.8f);

				cd_.composite_auto_weights_ = ptree.get<bool>("composite.auto_weights", false);

				// Display
				window_max_size = ptree.get<uint>("display.main_window_max_size", 768);
				time_filter_cuts_window_max_size = ptree.get<uint>("display.time_filter_cuts_window_max_size", 512);
				auxiliary_window_max_size = ptree.get<uint>("display.auxiliary_window_max_size", 512);

				notify();
			}
		}

		void MainWindow::save_ini(const std::string& path)
		{
			boost::property_tree::ptree ptree;
			GroupBox *image_rendering_group_box = ui.ImageRenderingGroupBox;
			GroupBox *view_group_box = ui.ViewGroupBox;
			GroupBox *record_group_box = ui.ExportGroupBox;
			GroupBox *import_group_box = ui.ImportGroupBox;
			GroupBox *info_group_box = ui.InfoGroupBox;
			Config& config = global::global_config;

			// Config
			ptree.put<uint>("config.file_buffer_size", config.file_buffer_size);
			ptree.put<uint>("config.input_buffer_size", config.input_queue_max_size);
			ptree.put<uint>("config.output_buffer_size", config.output_queue_max_size);
			ptree.put<uint>("config.time_filter_cuts_output_buffer_size", config.time_filter_cuts_output_buffer_size);
			ptree.put<uint>("config.accumulation_buffer_size", cd_.img_acc_slice_xy_level);
			ptree.put<uint>("config.frame_timeout", config.frame_timeout);
			ptree.put<bool>("config.flush_on_refresh", config.flush_on_refresh);
			ptree.put<ushort>("config.display_rate", static_cast<ushort>(cd_.display_rate));

			// Image rendering
			ptree.put<bool>("image_rendering.hidden", image_rendering_group_box->isHidden());
			ptree.put("image_rendering.camera", kCamera);
			ptree.put<ushort>("image_rendering.time_filter_size", cd_.time_filter_size);
			ptree.put<ushort>("image_rendering.p_index", cd_.pindex);
			ptree.put<float>("image_rendering.lambda", cd_.lambda);
			ptree.put<float>("image_rendering.z_distance", cd_.zdistance);
			ptree.put<double>("image_rendering.z_step", z_step_);
			ptree.put<holovibes::Algorithm>("image_rendering.algorithm", cd_.algorithm);
			ptree.put<ushort>("image_rendering.raw_bitshift", cd_.raw_bitshift);
			ptree.put<ushort>("image_rendering.time_filter_stride", cd_.time_filter_stride);

			// View
			ptree.put<bool>("view.hidden", view_group_box->isHidden());
			ptree.put<holovibes::ImgType>("view.view_mode", cd_.img_type);
			ptree.put<bool>("view.log_scale_enabled", cd_.log_scale_slice_xy_enabled);
			ptree.put<bool>("view.log_scale_enabled_cut_xz", cd_.log_scale_slice_xz_enabled);
			ptree.put<bool>("view.log_scale_enabled_cut_yz", cd_.log_scale_slice_yz_enabled);
			ptree.put<bool>("view.fft_shift_enabled", cd_.fft_shift_enabled);
			ptree.put<bool>("view.contrast_enabled", cd_.contrast_enabled);
			ptree.put<float>("view.contrast_threshold_low_percentile", cd_.contrast_threshold_low_percentile);
			ptree.put<float>("view.contrast_threshold_high_percentile", cd_.contrast_threshold_high_percentile);

			ptree.put<float>("view.contrast_min", cd_.contrast_min_slice_xy);
			ptree.put<float>("view.contrast_max", cd_.contrast_max_slice_xy);
			ptree.put<ushort>("view.cuts_contrast_p_offset", cd_.cuts_contrast_p_offset);
			ptree.put<bool>("view.accumulation_enabled", cd_.img_acc_slice_xy_enabled);
			ptree.put<float>("view.mainWindow_rotate", displayAngle);
			ptree.put<float>("view.xCut_rotate", xzAngle);
			ptree.put<float>("view.yCut_rotate", yzAngle);
			ptree.put<int>("view.mainWindow_flip", displayFlip);
			ptree.put<int>("view.xCut_flip", xzFlip);
			ptree.put<int>("view.yCut_flip", yzFlip);
			ptree.put<float>("view.reticle_scale", cd_.reticle_scale);

			// Chart
			ptree.put<size_t>("chart.auto_scale_point_threshold", auto_scale_point_threshold_);

			// Record
			ptree.put<bool>("record.hidden", record_group_box->isHidden());
			ptree.put<uint>("record.record_frame_step", record_frame_step_);

			// Import
			ptree.put<bool>("import.hidden", import_group_box->isHidden());
			ptree.put<float>("import.pixel_size", cd_.pixel_size);

			// Info
			ptree.put<bool>("info.hidden", info_group_box->isHidden());
			ptree.put<ushort>("info.theme_type", theme_index_);

			// Composite
			ptree.put<ushort>("composite.p_red", cd_.composite_p_red);
			ptree.put<ushort>("composite.p_blue", cd_.composite_p_blue);
			ptree.put<float>("composite.weight_r", cd_.weight_r);
			ptree.put<float>("composite.weight_g", cd_.weight_g);
			ptree.put<float>("composite.weight_b", cd_.weight_b);

			ptree.put<ushort>("composite.p_min_h", cd_.composite_p_min_h);
			ptree.put<ushort>("composite.p_max_h", cd_.composite_p_max_h);
			ptree.put<float>("composite.slider_h_threshold_min", cd_.slider_h_threshold_min);
			ptree.put<float>("composite.slider_h_threshold_max", cd_.slider_h_threshold_max);
			ptree.put<float>("composite.low_h_threshold", cd_.composite_low_h_threshold);
			ptree.put<float>("composite.high_h_threshold", cd_.composite_high_h_threshold);

			ptree.put<bool>("composite.p_activated_s", cd_.composite_p_activated_s);
			ptree.put<ushort>("composite.p_min_s", cd_.composite_p_min_s);
			ptree.put<ushort>("composite.p_max_s", cd_.composite_p_max_s);
			ptree.put<float>("composite.slider_s_threshold_min", cd_.slider_s_threshold_min);
			ptree.put<float>("composite.slider_s_threshold_max", cd_.slider_s_threshold_max);
			ptree.put<float>("composite.low_s_threshold", cd_.composite_low_s_threshold);
			ptree.put<float>("composite.high_s_threshold", cd_.composite_high_s_threshold);

			ptree.put<bool>("composite.p_activated_v", cd_.composite_p_activated_v);
			ptree.put<ushort>("composite.p_min_v", cd_.composite_p_min_v);
			ptree.put<ushort>("composite.p_max_v", cd_.composite_p_max_v);
			ptree.put<float>("composite.slider_v_threshold_min", cd_.slider_v_threshold_min);
			ptree.put<float>("composite.slider_v_threshold_max", cd_.slider_v_threshold_max);
			ptree.put<float>("composite.low_v_threshold", cd_.composite_low_v_threshold);
			ptree.put<float>("composite.high_v_threshold", cd_.composite_high_v_threshold);
			ptree.put<bool>("composite.auto_weights", cd_.composite_auto_weights_);

			//Reset
			ptree.put<bool>("reset.set_cuda_device", config.set_cuda_device);
			ptree.put<bool>("reset.auto_device_number", config.auto_device_number);
			ptree.put<uint>("reset.device_number", config.device_number);

			// Display
			ptree.put<uint>("display.main_window_max_size", window_max_size);
            ptree.put<uint>("display.time_filter_cuts_window_max_size", time_filter_cuts_window_max_size);
			ptree.put<uint>("display.auxiliary_window_max_size", auxiliary_window_max_size);

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
			if (cd_.convolution_enabled)
				set_convolution_mode(false);
			if (cd_.chart_enabled)
				disable_chart_mode();
			cancel_time_filter_cuts();
			if (cd_.filter_2d_enabled)
				cancel_filter2D();
			holovibes_.dispose_compute();
		}

		void MainWindow::camera_none()
		{
			close_windows();
			close_critical_compute();
			if (!is_raw_mode())
				holovibes_.dispose_compute();
			holovibes_.dispose_capture();
			remove_infos();
			ui.actionSettings->setEnabled(false);
			is_enabled_camera_ = false;
			cd_.compute_mode = Computation::Stop;
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
			cd_.gpu_lens_display_enabled = false;

			/* Raw view & recording */
			raw_window.reset(nullptr);
			cd_.raw_view = false;
			cd_.record_raw = false;
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
			if (!is_raw_mode())
				holovibes_.dispose_compute();
			holovibes_.dispose_capture();
			cd_.pindex = 0;
			cd_.time_filter_size = 1;
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
			if (cd_.compute_mode != Computation::Stop)
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
					if (!is_raw_mode())
						holovibes_.dispose_compute();
					holovibes_.dispose_capture();

					set_camera_timeout();

					//Needed for correct read of SquareInputMode during allocation of buffers
					set_computation_mode();
					set_correct_square_input_mode();

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

		void MainWindow::camera_hamamatsu()
		{
			change_camera(CameraKind::Hamamatsu);
		}

		void MainWindow::camera_adimec()
		{
			change_camera(CameraKind::Adimec);
		}

		void MainWindow::camera_xiq()
		{
			change_camera(CameraKind::xiQ);
		}

		void MainWindow::camera_xib()
		{
			change_camera(CameraKind::xiB);
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

		void MainWindow::set_raw_mode()
		{
			close_windows();
			close_critical_compute();
			ui.SquareInputModeComboBox->setEnabled(false);
			InfoManager::get_manager()->remove_info("Throughput");
			cd_.compute_mode = Computation::Stop;
			notify();
			if (is_enabled_camera_)
			{
				QPoint pos(0, 0);
				const FrameDescriptor& fd = holovibes_.get_gpu_input_queue()->get_fd();
				width = fd.width;
				height = fd.height;
				get_good_size(width, height, window_max_size);
				QSize size(width, height);
				init_image_mode(pos, size);
				cd_.compute_mode = Computation::Raw;
				createPipe();
				mainDisplay.reset(
					new RawWindow(
						pos, size,
						holovibes_.get_gpu_input_queue()));
				mainDisplay->setTitle(QString("XY view"));
				mainDisplay->setCd(&cd_);
				mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));
				InfoManager::get_manager()->insertFrameDescriptorInfo(fd, InfoManager::InfoType::INPUT_SOURCE, "Input Format");
				set_convolution_mode(false);
				set_divide_convolution_mode(false);
				notify();
				layout_toggled();
			}
		}

		void MainWindow::createPipe()
		{

			unsigned int depth = holovibes_.get_gpu_input_queue()->get_fd().depth;

			if (cd_.compute_mode == Computation::Hologram)
			{
				depth = 2;
				if (cd_.img_type == ImgType::Composite)
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
			const FrameDescriptor& fd = holovibes_.get_gpu_input_queue()->get_fd();
			width = fd.width;
			height = fd.height;
			get_good_size(width, height, window_max_size);
			QSize size(width, height);
			init_image_mode(pos, size);
			/* ---------- */
			try
			{
				mainDisplay.reset(
					new HoloWindow(
						pos, size,
						holovibes_.get_gpu_output_queue(),
						holovibes_.get_pipe(),
						sliceXZ,
						sliceYZ,
						this));
				mainDisplay->set_is_resize(false);
				mainDisplay->setTitle(QString("XY view"));
				mainDisplay->setCd(&cd_);
				mainDisplay->resetTransform();
				mainDisplay->setAngle(displayAngle);
				mainDisplay->setFlip(displayFlip);
				mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));

			}
			catch (std::runtime_error& e)
			{
				LOG_ERROR(std::string("createHoloWindow: ") + std::string(e.what()));
			}
		}

		void MainWindow::set_holographic_mode()
		{
			// That function is used to reallocate the buffers since the Square
			// input mode could have changed
			/* Close windows & destory thread compute */
			close_windows();
			close_critical_compute();

			ui.SquareInputModeComboBox->setEnabled(true);

			/* ---------- */
			try
			{
				cd_.compute_mode = Computation::Hologram;
				/* Pipe & Window */
				createPipe();
				createHoloWindow();
				/* Info Manager */
				const FrameDescriptor& fd = holovibes_.get_gpu_output_queue()->get_fd();
				InfoManager::get_manager()->insertFrameDescriptorInfo(fd, InfoManager::InfoType::OUTPUT_SOURCE, "Output format");
				/* Contrast */
				cd_.contrast_enabled = true;
				if (cd_.file_type != FileType::HOLO)
				{
					set_auto_contrast(); // Set auto contrast on the current window
					auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
					// Set auto contrast on the XY view even if it is not the current window
					if (pipe)
						pipe->autocontrast_end_pipe(XYview);
				}
				/* Notify */
				notify();
			}
			catch (std::runtime_error& e)
			{
				LOG_ERROR(std::string("cannot set holographic mode: ") + std::string(e.what()));
			}
		}

		void MainWindow::set_computation_mode()
		{
			if (ui.DirectRadioButton->isChecked())
			{
				cd_.compute_mode = Computation::Raw;
			}
			else if (ui.HologramRadioButton->isChecked())
			{
				cd_.compute_mode = Computation::Hologram;
			}
		}

		void MainWindow::set_correct_square_input_mode()
		{
			if (cd_.compute_mode == Computation::Raw)
			{
				cd_.square_input_mode = SquareInputMode::NO_MODIFICATION;
			}
			else if (cd_.compute_mode == Computation::Hologram)
			{
				cd_.square_input_mode = get_square_input_mode_from_string(ui.SquareInputModeComboBox->currentText().toStdString());
			}
		}

		void MainWindow::set_camera_timeout()
		{
			camera::FRAME_TIMEOUT = global::global_config.frame_timeout;
		}

		void MainWindow::set_square_input_mode(const QString &name)
		{
			auto mode = get_square_input_mode_from_string(name.toStdString());
			cd_.square_input_mode = mode;
			//Need to reset the whole computation process since we change the size of the different buffers
			reset_input();
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
			if (!is_raw_mode())
			{
				QComboBox* ptr = ui.ViewModeComboBox;

				if (need_refresh(last_img_type_, value))
				{
				// This crash in debug mode, but surprinsingly, it works perfectly in release mode.
					cd_.img_type = static_cast<ImgType>(ptr->currentIndex());
					refreshViewMode();
					if (cd_.img_type == ImgType::Composite)
					{
						const unsigned min_val_composite = cd_.time_filter_size == 1 ? 0 : 1;
						const unsigned max_val_composite = cd_.time_filter_size - 1;

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

				pipe->insert_fn_end_vect([=]() {
					cd_.img_type = static_cast<ImgType>(ptr->currentIndex());
					notify();
					layout_toggled();
				});
				pipe_refresh();

				// Force XYview autocontrast
				pipe->autocontrast_end_pipe(XYview);
				// Force cuts views autocontrast if needed
				if (cd_.time_filter_cuts_enabled)
					set_auto_contrast_cuts();
			}
		}

		bool MainWindow::is_raw_mode()
		{
			return cd_.compute_mode == Computation::Raw;
		}

		void MainWindow::set_image_mode()
		{
			if (cd_.compute_mode == Computation::Raw)
				set_raw_mode();
			else if (cd_.compute_mode == Computation::Hologram)
				set_holographic_mode();
			else
			{
			if (ui.DirectRadioButton->isChecked())
					set_raw_mode();
				else
					set_holographic_mode();
			}
		}
#pragma endregion

#pragma region Batch

		static void adapt_time_filter_stride_to_batch_size(ComputeDescriptor& cd)
		{
			if (cd.time_filter_stride < cd.batch_size)
				cd.time_filter_stride = cd.batch_size;
			// Go to lower multiple
			if (cd.time_filter_stride % cd.batch_size != 0)
				cd.time_filter_stride -= cd.time_filter_stride % cd.batch_size;
		}

		void MainWindow::update_batch_size()
		{
			if (!is_raw_mode())
			{
				int value = ui.BatchSizeSpinBox->value();

				if (value == cd_.batch_size)
					return;

				auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
				if (pipe)
				{
					pipe->insert_fn_end_vect([=]() {
						cd_.batch_size = value;
						adapt_time_filter_stride_to_batch_size(cd_);
						holovibes_.get_pipe()->request_update_batch_size();
						notify();
					});
				}
				else
					std::cout << "COULD NOT GET PIPE" << std::endl;
			}
		}

#pragma endregion
		/* ------------ */
#pragma region STFT
		void MainWindow::cancel_stft_slice_view()
		{
			InfoManager *manager = InfoManager::get_manager();

			manager->remove_info("STFT Slice Cursor");

			cd_.contrast_max_slice_xz = false;
			cd_.contrast_max_slice_yz = false;
			cd_.log_scale_slice_xz_enabled = false;
			cd_.log_scale_slice_yz_enabled = false;
			cd_.img_acc_slice_xz_enabled = false;
			cd_.img_acc_slice_yz_enabled = false;
			sliceXZ.reset(nullptr);
			sliceYZ.reset(nullptr);

			if (mainDisplay)
			{
				mainDisplay->setCursor(Qt::ArrowCursor);
				mainDisplay->getOverlayManager().disable_all(SliceCross);
				mainDisplay->getOverlayManager().disable_all(Cross);
			}
			if (auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get()))
			{
				pipe->insert_fn_end_vect([=]() {
					cd_.time_filter_cuts_enabled = false;
					pipe->delete_stft_slice_queue();

					ui.TimeFilterCutsCheckBox->setChecked(false);
					notify();
				});
			}

		}

		void MainWindow::update_time_filter_stride()
		{
			if (!is_raw_mode())
			{
				int value = ui.STFTStepsSpinBox->value();

				if (value == cd_.time_filter_stride)
					return;

				auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
				if (pipe)
				{
					pipe->insert_fn_end_vect([=]() {
						cd_.time_filter_stride = value;
						adapt_time_filter_stride_to_batch_size(cd_);
						holovibes_.get_pipe()->request_update_time_filter_stride();
						notify();
					});
				}
				else
					std::cout << "COULD NOT GET PIPE" << std::endl;
			}
		}

		void MainWindow::toggle_time_filter_cuts(bool checked)
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
					if (cd_.filter_2d_enabled)
						cancel_filter2D();
					holovibes_.get_pipe()->create_stft_slice_queue();
					// set positions of new windows according to the position of the main GL window
					QPoint			xzPos = mainDisplay->framePosition() + QPoint(0, mainDisplay->height() + 42);
					QPoint			yzPos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 20, 0);
					const ushort	nImg = cd_.time_filter_size;
					uint			time_filter_size = std::max(256u, std::min(512u, (uint)nImg));

					if (time_filter_size > time_filter_cuts_window_max_size)
						time_filter_size = time_filter_cuts_window_max_size;

					while (holovibes_.get_pipe()->get_update_time_filter_size_request());
					while (holovibes_.get_pipe()->get_cuts_request());
					sliceXZ.reset(new SliceWindow(
						xzPos,
						QSize(mainDisplay->width(), time_filter_size),
						holovibes_.get_pipe()->get_stft_slice_queue(0),
						KindOfView::SliceXZ,
						this));
					sliceXZ->setTitle("XZ view");
					sliceXZ->setAngle(xzAngle);
					sliceXZ->setFlip(xzFlip);
					sliceXZ->setCd(&cd_);

					sliceYZ.reset(new SliceWindow(
						yzPos,
						QSize(time_filter_size, mainDisplay->height()),
						holovibes_.get_pipe()->get_stft_slice_queue(1),
						KindOfView::SliceYZ,
						this));
					sliceYZ->setTitle("YZ view");
					sliceYZ->setAngle(yzAngle);
					sliceYZ->setFlip(yzFlip);
					sliceYZ->setCd(&cd_);

					mainDisplay->getOverlayManager().create_overlay<Cross>();
					cd_.time_filter_cuts_enabled = true;
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

		void MainWindow::cancel_time_filter_cuts()
		{
			if (cd_.time_filter_cuts_enabled)
				cancel_stft_slice_view();
			try {
				while (holovibes_.get_pipe()->get_refresh_request());
			}
			catch (std::exception&)
			{
			}
			cd_.time_filter_cuts_enabled = false;
			notify();
		}

#pragma endregion
		/* ------------ */
#pragma region Computation
		void MainWindow::change_window()
		{
			QComboBox *window_cbox = ui.WindowSelectionComboBox;

			if (window_cbox->currentIndex() == 0)
				cd_.current_window = WindowKind::XYview;
			else if (window_cbox->currentIndex() == 1)
				cd_.current_window = WindowKind::XZview;
			else if (window_cbox->currentIndex() == 2)
				cd_.current_window = WindowKind::YZview;
			pipe_refresh();
			notify();
		}

		void MainWindow::load_convo_matrix()
		{
			holovibes_.clear_convolution_matrix();

			try
			{
				std::filesystem::path dir(get_exe_dir());
				dir = dir / "convolution_kernels" / ui.KernelQuickSelectComboBox->currentText().toStdString();
				std::string path = dir.string();

				std::vector<float> matrix;
				unsigned matrix_width = 0;
				unsigned matrix_height = 0;
				unsigned matrix_z = 1;

				// Doing this the C way cause it's faster
				FILE* c_file;
				fopen_s(&c_file, path.c_str(), "r");

				if (c_file == nullptr)
				{
					fclose(c_file);
					throw std::runtime_error("Invalid file path");
				}

				// Read kernel dimensions
				if (fscanf_s(c_file, "%u %u %u;", &matrix_width, &matrix_height, &matrix_z) != 3)
				{
					fclose(c_file);
					throw std::runtime_error("Invalid kernel dimensions");
				}

				size_t matrix_size = matrix_width * matrix_height * matrix_z;
				matrix.resize(matrix_size);

				// Read kernel values
				for (size_t i = 0; i < matrix_size; ++i)
				{
					if (fscanf_s(c_file, "%f", &matrix[i]) != 1)
					{
						fclose(c_file);
						throw std::runtime_error("Missing values");
					}
				}

				fclose(c_file);

				//on plonge le kernel dans un carre de taille nx*ny tout en gardant le profondeur z
				uint c = 0;
				uint nx = holovibes_.get_gpu_output_queue()->get_fd().width;
				uint ny = holovibes_.get_gpu_output_queue()->get_fd().height;
				uint size = nx * ny;

				const uint minw = (nx / 2) - (matrix_width / 2);
				const uint maxw = (nx / 2) + (matrix_width / 2);
				const uint minh = (ny / 2) - (matrix_height / 2);
				const uint maxh = (ny / 2) + (matrix_height / 2);

				std::vector<float> convo_matrix(size, 0.0f);

				for (size_t i = minh; i < maxh; i++)
				{
					for (size_t j = minw; j < maxw; j++)
					{
						convo_matrix[i * nx + j] = matrix[c];
						c++;
					}
				}

				//on met les largeurs et hauteurs a la taille de nx et de ny
				cd_.convo_matrix_width = nx;
				cd_.convo_matrix_height = ny;
				cd_.convo_matrix_z = matrix_z;
				cd_.convo_matrix = convo_matrix;
			}
			catch (std::exception& e)
			{
				holovibes_.clear_convolution_matrix();
				display_error("Couldn't load file\n" + std::string(e.what()));
			}
		}

		void MainWindow::set_convolution_mode(const bool value)
		{
			if (!value && cd_.convolution_enabled)
				set_divide_convolution_mode(false);

			load_convo_matrix();

			cd_.convolution_changed = cd_.convolution_enabled != value;
			cd_.convolution_enabled = value;

			pipe_refresh();
			notify();
		}

		void MainWindow::set_divide_convolution_mode(const bool value)
		{
			cd_.divide_convolution_enabled = value;

			pipe_refresh();
			notify();
		}

		void MainWindow::toggle_renormalize(bool value)
		{
			cd_.renorm_enabled = value;
			pipe_refresh();
			notify();
		}

		void MainWindow::set_renormalize_constant(int value)
		{
			cd_.renorm_constant = value;
		}

		void MainWindow::set_filter2D()
		{
			if (!is_raw_mode())
			{
				mainDisplay->resetTransform();
				mainDisplay->getOverlayManager().create_overlay<Filter2D>();
				ui.Filter2DPushButton->setStyleSheet("QPushButton {color: #009FFF;}");
				cd_.log_scale_slice_xy_enabled = true;
				cd_.fft_shift_enabled = true;
				cd_.filter_2d_enabled = true;
				if (auto pipe = dynamic_cast<Pipe*>(holovibes_.get_pipe().get()))
					pipe->autocontrast_end_pipe(XYview);
				InfoManager::get_manager()->update_info("Filter2D", "Processing...");
				notify();
			}
		}

		void MainWindow::set_filter2D_type(const QString &filter2Dtype)
		{
			const std::string &type_str = filter2Dtype.toStdString();
			Filter2DType type = Filter2DType::LowPass;
			if (type_str == "Low pass")
			{
				type = Filter2DType::LowPass;
			}
			else if (type_str == "High pass")
			{
				type = Filter2DType::HighPass;
			}
			else if (type_str == "Band pass")
			{
				type = Filter2DType::BandPass;
			}

			cd_.filter_2d_type = type;

			notify();
		}

		void MainWindow::cancel_filter2D()
		{
			if (!is_raw_mode())
			{
				InfoManager::get_manager()->remove_info("Filter2D");
				cd_.filter_2d_enabled = false;
				cd_.log_scale_slice_xy_enabled = false;
				auto zone = units::RectFd({0, 0}, {0, 0});
				cd_.setStftZone(zone);
				cd_.setFilter2DSubZone(zone);
				if (mainDisplay)
				{
					mainDisplay->getOverlayManager().disable_all(Filter2D);
					mainDisplay->getOverlayManager().create_default();
					mainDisplay->resetTransform();
				}
				set_auto_contrast();
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_fft_shift(const bool value)
		{
			if (!is_raw_mode())
			{
				cd_.fft_shift_enabled = value;
				pipe_refresh();
			}
		}

		void MainWindow::set_time_filter_size()
		{
			if (!is_raw_mode())
			{
				int time_filter_size = ui.timeFilterSizeSpinBox->value();
				time_filter_size = std::max(1, time_filter_size);

				if (time_filter_size == cd_.time_filter_size)
					return;
				notify();
				auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get());
				if (pipe)
				{
					pipe->insert_fn_end_vect([=]() {
						cd_.time_filter_size = time_filter_size;
						holovibes_.get_pipe()->request_update_time_filter_size();
						set_p_accu();
						// This will not do anything until SliceWindow::changeTexture() isn't coded.
					});
				}
			}
		}

		void MainWindow::update_lens_view(bool value)
		{
			cd_.gpu_lens_display_enabled = value;

			if (value)
			{
				try
				{
					// set positions of new windows according to the position of the main GL window
					QPoint pos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 310, 0);
					ICompute* pipe = holovibes_.get_pipe().get();

					ushort lens_window_width = width;
					ushort lens_window_height = height;
					get_good_size(lens_window_width, lens_window_height, auxiliary_window_max_size);

					lens_window.reset(new RawWindow(pos,
													QSize(lens_window_width, lens_window_height),
													pipe->get_lens_queue(),
													KindOfView::Lens));

					lens_window->setTitle("Lens view");
					lens_window->setCd(&cd_);

					connect(lens_window.get(), SIGNAL(destroyed()), this, SLOT(disable_lens_view()));
				}
				catch (std::exception& e)
				{
					std::cerr << e.what() << std::endl;
				}
			}

			else
			{
				disable_lens_view();
				lens_window.reset(nullptr);
			}

			pipe_refresh();
		}

		void MainWindow::disable_lens_view()
		{
			if (lens_window)
				disconnect(lens_window.get(), SIGNAL(destroyed()), this, SLOT(disable_lens_view()));

			cd_.gpu_lens_display_enabled = false;
			holovibes_.get_pipe()->request_disable_lens_view();
			notify();
		}

		void MainWindow::update_raw_view(bool value)
		{

			if (value)
			{
				ICompute* pipe = holovibes_.get_pipe().get();
				if (!pipe->is_raw_queue_allocated())
					pipe->request_allocate_raw_queue();
				// Wait until the pipe has been allocated and ready to use
				while (!pipe->is_raw_queue_allocated());

				cd_.raw_view = true;

				ushort raw_window_width = width;
				ushort raw_window_height = height;
				get_good_size(raw_window_width, raw_window_height, auxiliary_window_max_size);

				// set positions of new windows according to the position of the main GL window and Lens window
				QPoint pos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 310, 0);
					raw_window.reset(new RawWindow(
						pos,
						QSize(raw_window_width, raw_window_height),
						pipe->get_raw_queue()));

				raw_window->setTitle("Raw view");
				raw_window->setCd(&cd_);

				connect(raw_window.get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));
			}
			else
			{
				cd_.raw_view = false;
				disable_raw_view();
				raw_window.reset(nullptr);
			}
			pipe_refresh();
		}

		void MainWindow::disable_raw_view()
		{
			if (raw_window)
				disconnect(raw_window.get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));

			cd_.raw_view = false;

			if (!cd_.record_raw)
			{
				gui::InfoManager::get_manager()->remove_info("RawOutputQueue");
				holovibes_.get_pipe()->request_kill_raw_queue();
			}

			notify();
		}

		void MainWindow::set_p_accu()
		{
			auto spinbox = ui.PAccSpinBox;
			auto checkBox = ui.PAccuCheckBox;
			cd_.p_accu_enabled = checkBox->isChecked();
			cd_.p_acc_level = spinbox->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_x_accu()
		{
			auto box = ui.XAccSpinBox;
			auto checkBox = ui.XAccuCheckBox;
			cd_.x_accu_enabled = checkBox->isChecked();
			cd_.x_acc_level = box->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_y_accu()
		{
			auto box = ui.YAccSpinBox;
			auto checkBox = ui.YAccuCheckBox;
			cd_.y_accu_enabled = checkBox->isChecked();
			cd_.y_acc_level = box->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_p(int value)
		{
			if (!is_raw_mode())
			{
				if (value < static_cast<int>(cd_.time_filter_size))
				{
					cd_.pindex = value;
					pipe_refresh();
					notify();
				}
				else
					display_error("p param has to be between 1 and #img");
			}
		}

		void MainWindow::set_composite_intervals()
		{
			ui.PRedSpinBox_Composite->setValue(std::min(ui.PRedSpinBox_Composite->value(), ui.PBlueSpinBox_Composite->value()));
			cd_.composite_p_red = ui.PRedSpinBox_Composite->value();
			cd_.composite_p_blue = ui.PBlueSpinBox_Composite->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_composite_intervals_hsv_h_min()
		{
			cd_.composite_p_min_h = ui.SpinBox_hue_freq_min->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_composite_intervals_hsv_h_max()
		{
			cd_.composite_p_max_h = ui.SpinBox_hue_freq_max->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_composite_intervals_hsv_s_min()
		{
			cd_.composite_p_min_s = ui.SpinBox_saturation_freq_min->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_composite_intervals_hsv_s_max()
		{
			cd_.composite_p_max_s = ui.SpinBox_saturation_freq_max->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_composite_intervals_hsv_v_min()
		{
			cd_.composite_p_min_v = ui.SpinBox_value_freq_min->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_composite_intervals_hsv_v_max()
		{
			cd_.composite_p_max_v = ui.SpinBox_value_freq_max->value();
			pipe_refresh();
			notify();
		}

		void MainWindow::set_composite_weights()
		{
			cd_.weight_r = ui.WeightSpinBox_R->value();
			cd_.weight_g = ui.WeightSpinBox_G->value();
			cd_.weight_b = ui.WeightSpinBox_B->value();
			pipe_refresh();
			notify();
		}


		void MainWindow::set_composite_auto_weights(bool value)
		{
			cd_.composite_auto_weights_ = value;
			set_auto_contrast();
		}

		void MainWindow::click_composite_rgb_or_hsv()
		{
			cd_.composite_kind = ui.radioButton_rgb->isChecked() ? CompositeKind::RGB : CompositeKind::HSV;
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
			cd_.composite_p_activated_s = ui.checkBox_saturation_freq->isChecked();
			ui.SpinBox_saturation_freq_min->setDisabled(!ui.checkBox_saturation_freq->isChecked());
			ui.SpinBox_saturation_freq_max->setDisabled(!ui.checkBox_saturation_freq->isChecked());
		}

		void MainWindow::actualize_frequency_channel_v()
		{
			cd_.composite_p_activated_v = ui.checkBox_value_freq->isChecked();
			ui.SpinBox_value_freq_min->setDisabled(!ui.checkBox_value_freq->isChecked());
			ui.SpinBox_value_freq_max->setDisabled(!ui.checkBox_value_freq->isChecked());
		}

		void MainWindow::actualize_checkbox_h_gaussian_blur()
		{
			cd_.h_blur_activated = ui.checkBox_h_gaussian_blur->isChecked();
			ui.SpinBox_hue_blur_kernel_size->setEnabled(ui.checkBox_h_gaussian_blur->isChecked());
		}

		void MainWindow::actualize_kernel_size_blur()
		{
			cd_.h_blur_kernel_size = ui.SpinBox_hue_blur_kernel_size->value();
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
			slide_update_threshold(*ui.horizontalSlider_hue_threshold_min, cd_.slider_h_threshold_min,
				cd_.slider_h_threshold_max, *ui.horizontalSlider_hue_threshold_max,
				*ui.label_hue_threshold_min, cd_.slider_h_threshold_min, cd_.slider_h_threshold_max);
		}

		void  MainWindow::slide_update_threshold_h_max()
		{
			slide_update_threshold(*ui.horizontalSlider_hue_threshold_max, cd_.slider_h_threshold_max,
				cd_.slider_h_threshold_min, *ui.horizontalSlider_hue_threshold_min,
				*ui.label_hue_threshold_max, cd_.slider_h_threshold_min, cd_.slider_h_threshold_max);
		}

		void MainWindow::slide_update_threshold_s_min()
		{
			slide_update_threshold(*ui.horizontalSlider_saturation_threshold_min, cd_.slider_s_threshold_min,
				cd_.slider_s_threshold_max, *ui.horizontalSlider_saturation_threshold_max,
				*ui.label_saturation_threshold_min, cd_.slider_s_threshold_min, cd_.slider_s_threshold_max);
		}

		void MainWindow::slide_update_threshold_s_max()
		{
			slide_update_threshold(*ui.horizontalSlider_saturation_threshold_max, cd_.slider_s_threshold_max,
				cd_.slider_s_threshold_min, *ui.horizontalSlider_saturation_threshold_min,
				*ui.label_saturation_threshold_max, cd_.slider_s_threshold_min, cd_.slider_s_threshold_max);
		}

		void MainWindow::slide_update_threshold_v_min()
		{
			slide_update_threshold(*ui.horizontalSlider_value_threshold_min, cd_.slider_v_threshold_min,
				cd_.slider_v_threshold_max, *ui.horizontalSlider_value_threshold_max,
				*ui.label_value_threshold_min, cd_.slider_v_threshold_min, cd_.slider_v_threshold_max);
		}

		void MainWindow::slide_update_threshold_v_max()
		{
			slide_update_threshold(*ui.horizontalSlider_value_threshold_max, cd_.slider_v_threshold_max,
				cd_.slider_v_threshold_min, *ui.horizontalSlider_value_threshold_min,
				*ui.label_value_threshold_max, cd_.slider_v_threshold_min, cd_.slider_v_threshold_max);
		}

		void MainWindow::increment_p()
		{
			if (!is_raw_mode())
			{

				if (cd_.pindex < cd_.time_filter_size)
				{
					cd_.pindex = cd_.pindex + 1;
					set_auto_contrast();
					notify();
				}
				else
					display_error("p param has to be between 1 and #img");
			}
		}

		void MainWindow::decrement_p()
		{
			if (!is_raw_mode())
			{
				if (cd_.pindex > 0)
				{
					cd_.pindex = cd_.pindex - 1;
					set_auto_contrast();
					notify();
				}
				else
					display_error("p param has to be between 1 and #img");
			}
		}

		void MainWindow::set_wavelength(const double value)
		{
			if (!is_raw_mode())
			{
				cd_.lambda = static_cast<float>(value) * 1.0e-9f;
				pipe_refresh();
			}
		}

		void MainWindow::set_z(const double value)
		{
			if (!is_raw_mode())
			{
				cd_.zdistance = static_cast<float>(value);
				pipe_refresh();
			}
		}

		void MainWindow::increment_z()
		{
			if (!is_raw_mode())
			{
				set_z(cd_.zdistance + z_step_);
				ui.ZDoubleSpinBox->setValue(cd_.zdistance);
			}
		}

		void MainWindow::decrement_z()
		{
			if (!is_raw_mode())
			{
				set_z(cd_.zdistance - z_step_);
				ui.ZDoubleSpinBox->setValue(cd_.zdistance);
			}
		}

		void MainWindow::set_z_step(const double value)
		{
			z_step_ = value;
			ui.ZDoubleSpinBox->setSingleStep(value);
		}

		void MainWindow::set_algorithm(const QString value)
		{
			if (!is_raw_mode())
			{
				if (value == "None")
					cd_.algorithm = Algorithm::None;
				else if (value == "1FFT")
					cd_.algorithm = Algorithm::FFT1;
				else if (value == "2FFT")
					cd_.algorithm = Algorithm::FFT2;
				else
				{
					// Shouldn't happen
					cd_.algorithm = Algorithm::None;
					LOG_ERROR("Unknown algorithm: " + value.toStdString() + ", falling back to None");
				}
				set_holographic_mode();
			}
		}

		void MainWindow::set_time_filter(QString value)
		{
			if (!is_raw_mode())
			{
				if (value == "STFT")
					cd_.time_filter = TimeFilter::STFT;
				else if (value == "PCA")
					cd_.time_filter = TimeFilter::PCA;
				set_holographic_mode();
			}
		}

		void MainWindow::set_unwrap_history_size(int value)
		{
			if (!is_raw_mode())
			{
				cd_.unwrap_history_size = value;
				holovibes_.get_pipe()->request_update_unwrap_size(value);
				notify();
			}
		}

		void MainWindow::set_unwrapping_1d(const bool value)
		{
			if (!is_raw_mode())
			{
				holovibes_.get_pipe()->request_unwrapping_1d(value);
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_unwrapping_2d(const bool value)
		{
			if (!is_raw_mode())
			{
				holovibes_.get_pipe()->request_unwrapping_2d(value);
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_accumulation(bool value)
		{
			if (!is_raw_mode())
			{
				cd_.set_accumulation(cd_.current_window, value);
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_accumulation_level(int value)
		{
			if (!is_raw_mode())
			{
				cd_.set_accumulation_level(cd_.current_window, value);
				pipe_refresh();
			}
		}

		void MainWindow::pipe_refresh()
		{
			if (!is_raw_mode())
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

		void MainWindow::set_composite_area()
		{
			mainDisplay->getOverlayManager().create_overlay<CompositeArea>();
		}

#pragma endregion
		/* ------------ */
#pragma region Texture
		void MainWindow::rotateTexture()
		{
			WindowKind curWin = cd_.current_window;

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
			WindowKind curWin = cd_.current_window;

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
#pragma region Contrast - Log
		void MainWindow::set_contrast_mode(bool value)
		{
			if (!is_raw_mode())
			{
				change_window();
				cd_.contrast_enabled = value;
				cd_.contrast_auto_refresh = true;
				pipe_refresh();
				notify();
			}
		}

		void MainWindow::set_auto_contrast_cuts()
		{
			if (auto pipe = dynamic_cast<Pipe *>(holovibes_.get_pipe().get()))
			{
				pipe->autocontrast_end_pipe(XZview);
				pipe->autocontrast_end_pipe(YZview);
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

		void MainWindow::set_auto_contrast()
		{
			if (!is_raw_mode())
			{
				try
				{
					if (auto pipe = dynamic_cast<Pipe*>(holovibes_.get_pipe().get()))
						pipe->autocontrast_end_pipe(cd_.current_window);
				}
				catch (std::runtime_error& e)
				{
					std::cerr << e.what() << std::endl;
				}
			}
		}

		void MainWindow::set_contrast_min(const double value)
		{
			if (!is_raw_mode())
			{
				if (cd_.contrast_enabled)
				{
					// Get the minimum contrast value rounded for the comparison
					const float old_val =
						cd_.get_truncate_contrast_min(cd_.current_window);
					// Floating number issue: cast to float for the comparison
					const float val = value;
					if (old_val != val)
					{
						cd_.set_contrast_min(cd_.current_window, value);
						pipe_refresh();
						notify();
					}
				}
			}
		}

		void MainWindow::set_contrast_max(const double value)
		{
			if (!is_raw_mode())
			{
				if (cd_.contrast_enabled)
				{
					// Get the maximum contrast value rounded for the comparison
					const float old_val =
						cd_.get_truncate_contrast_max(cd_.current_window);
					 // Floating number issue: cast to float for the comparison
					const float val = value;
					if (old_val != val)
					{
						cd_.set_contrast_max(cd_.current_window, value);
						pipe_refresh();
						notify();
					}
				}
			}
		}

		void MainWindow::invert_contrast(bool value)
		{
			if (!is_raw_mode())
			{
				if (cd_.contrast_enabled)
				{
					cd_.contrast_invert = value;
					pipe_refresh();
				}
			}
		}

		void MainWindow::set_auto_refresh_contrast(bool value)
		{
			cd_.contrast_auto_refresh = value;
			pipe_refresh();
			notify();
		}

		void MainWindow::set_log_scale(const bool value)
		{
			if (!is_raw_mode())
			{
				cd_.set_log_scale_slice_enabled(cd_.current_window, value);
				pipe_refresh();
				notify();
			}
		}
#pragma endregion
		/* ------------ */
#pragma region Chart

		void MainWindow::set_chart_mode(const bool value)
		{
			cd_.chart_enabled = value;

			if (mainDisplay)
			{
				mainDisplay->resetTransform();
				if (value)
					mainDisplay->getOverlayManager().create_overlay<Signal>();
				else
					disable_chart_mode();

				notify();
			}
		}

		void MainWindow::disable_chart_mode()
		{
			cd_.chart_enabled = false;

			if (mainDisplay)
			{
				mainDisplay->resetTransform();

				mainDisplay->getOverlayManager().disable_all(Signal);
				mainDisplay->getOverlayManager().disable_all(Noise);
			}

			holovibes_.get_pipe()->request_chart_stop();

			holovibes_.get_chart_queue().clear();
			plot_window_.reset(nullptr);

			pipe_refresh();

			notify();
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

		void MainWindow::set_chart_graphic()
		{
			if (plot_window_ != nullptr)
				return;

			PlotWindow *plot_window = new PlotWindow(holovibes_.get_chart_queue(), auto_scale_point_threshold_, "ROI Chart");

			connect(plot_window, SIGNAL(closed()), this, SLOT(disable_chart_mode()), Qt::UniqueConnection);
			holovibes_.get_pipe()->request_chart(&holovibes_.get_chart_queue());

			pipe_refresh();
			plot_window_.reset(plot_window);
		}

		void MainWindow::browse_roi_output_file()
		{
			QString filename = QFileDialog::getSaveFileName(this,
				tr("ROI output file"), "C://", tr("Text files (*.txt);;CSV files (*.csv)"));

			QLineEdit* roi_output_line_edit = ui.ROIOutputPathLineEdit;
			roi_output_line_edit->clear();
			roi_output_line_edit->insert(filename);
		}

		void MainWindow::chart_record()
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
			if (nb_frames_ == 0)
				return;
			std::string output_path = output_line_edit->text().toUtf8();
			if (output_path == "")
			{
				roi_stop_push_button->setDisabled(true);
				return display_error("No output file");
			}

			CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
				holovibes_.get_chart_queue(),
				output_path,
				nb_frames_,
				this));
			connect(CSV_record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_chart_record()));
			CSV_record_thread_->start();

			roi_stop_push_button->setDisabled(false);
		}

		void MainWindow::finished_chart_record()
		{
			CSV_record_thread_.reset(nullptr);
			display_info("ROI record done");

			QPushButton* roi_stop_push_button = ui.ROIOutputStopPushButton;
			roi_stop_push_button->setDisabled(true);
		}
#pragma endregion
		/* ------------ */
#pragma region Convolution
		void MainWindow::update_convo_kernel(const QString& value)
		{
			if (cd_.convolution_enabled)
			{
				load_convo_matrix();

				cd_.convolution_changed = true;

				pipe_refresh();
				notify();
			}
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

		std::string MainWindow::set_record_filename_properties(FrameDescriptor fd, std::string filename, bool add_info)
		{
			std::string sub_str = "";

			if (add_info)
			{
				std::string slice;
				switch (cd_.current_window)
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
				std::string mode = (is_raw_mode() || cd_.record_raw) ? "D" : "H";

				int depth = fd.depth;
				if (depth == 6)
					depth = 3;

				sub_str = "_" + slice +
						  "_" + mode +
						  "_" + std::to_string(fd.width) +
						  "_" + std::to_string(fd.height) +
						  "_" + std::to_string(depth << 3) + "bit_e";
			}

			// Insert sub_str before extension (or at the end if no extension)
			size_t dot_index = filename.find_last_of('.');
			if (dot_index == filename.npos)
				dot_index = filename.size();
			filename.insert(dot_index, sub_str, 0, sub_str.length());

			// Make sure 2 files don't have the same name by adding -1 / -2 / -3 ... in the name
			unsigned i = 1;
			while (std::filesystem::exists(filename))
			{
				if (i == 1)
				{
					filename.insert(dot_index, "-1", 0, 2);
					++i;
					continue;
				}
				unsigned digits_nb = std::log10(i - 1) + 1;
				filename.replace(dot_index, digits_nb + 1, "-" + std::to_string(i));
				++i;
			}

			return filename;
		}

		void MainWindow::set_raw_recording(bool value)
		{
			ICompute* pipe = holovibes_.get_pipe().get();

			// When switching to raw recording, we no longer care about
			// having a big Pipe::output_ buffer for the processed output,
			// and we need GPU memory for the Pipe::gpu_raw_queue_ so that
			// we don't miss any frame.
			if (value)
			{
				// Use an output Queue of size 4
				pipe->request_resize(4);
				if (!pipe->is_raw_queue_allocated())
					pipe->request_allocate_raw_queue();
				// We wait the request to be finished on the creation of the
				// thread reader
				cd_.record_raw = true;
			}
			else
			{
				// Restore original size
				pipe->request_resize(global::global_config.output_queue_max_size);
				if (!cd_.raw_view)
					pipe->request_kill_raw_queue();
				cd_.record_raw = false;
			}
		}

		void MainWindow::set_synchronized_record(bool value)
		{
			cd_.synchronized_record = value;
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
			cd_.reticle_scale = value;
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

			cd_.nb_frames_record = nb_of_frames_spinbox->value();
			if (cd_.nb_frames_record == 0)
				return;

			std::string path = path_line_edit->text().toUtf8();
			if (path == "")
				return display_error("No output file");

			try
			{
				Queue *queue;
				if (cd_.record_raw)
				{
					// Wait until the raw queue has been allocated by the pipe
					// (thread compute) and ready to use.
					ICompute* pipe = holovibes_.get_pipe().get();
					while (!pipe->is_raw_queue_allocated());
					queue = pipe->get_raw_queue().get();
					queue->set_display(true);
				}
				else
					queue = holovibes_.get_current_window_output_queue().get();

				if (queue)
				{
					path = set_record_filename_properties(queue->get_fd(), path, false);
					record_thread_.reset(new ThreadRecorder(*queue, path, holo_file_get_json_settings(queue), cd_, this));

					connect(record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_image_record()));
					if (cd_.synchronized_record)
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
					// Record button
					ui.ImageOutputRecPushButton->setEnabled(false);
					ui.ImageOutputBatchPushButton->setEnabled(false);

				}
				else
					throw std::exception("Unable to launch record");
			}
			catch (std::exception& e)
			{
				display_error(e.what());
			}
		}

		void MainWindow::set_record_frame_step(int value)
		{
			record_frame_step_ = value;
			ui.NumberOfFramesSpinBox->setSingleStep(value);
		}

		void MainWindow::finished_image_record()
		{
			QProgressBar* progress_bar = InfoManager::get_manager()->get_progress_bar();

			ui.ImageOutputStopPushButton->setDisabled(true);
			ui.ImageOutputRecPushButton->setEnabled(true);
			ui.ImageOutputBatchPushButton->setEnabled(true);

			if (cd_.record_raw && !cd_.raw_view)
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
			cd_.nb_frames_record = frame_nb_spin_box->value();
			std::string formatted_path;

			try
			{
				Queue* q = nullptr;

				if (cd_.current_window == WindowKind::XYview)
					q = holovibes_.get_gpu_output_queue().get();
				else if (cd_.current_window == WindowKind::XZview)
					q = holovibes_.get_pipe()->get_stft_slice_queue(0).get();
				else
					q = holovibes_.get_pipe()->get_stft_slice_queue(1).get();
				// Only loading the dll at runtime
				gpib_interface_ = gpib::GpibDLL::load_gpib("gpib.dll", input_path);

				formatted_path = format_batch_output(path, file_index_);
				formatted_path = set_record_filename_properties(q->get_fd(), formatted_path, false);

				if (gpib_interface_->execute_next_block()) // More blocks to come, use batch_next_block method.
				{
					if (is_batch_img_)
					{
						record_thread_.reset(new ThreadRecorder(*q, formatted_path, holo_file_get_json_settings(q), cd_, this));
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
							holovibes_.get_chart_queue(),
							formatted_path,
							cd_.nb_frames_record,
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
						record_thread_.reset(new ThreadRecorder(*q, formatted_path, holo_file_get_json_settings(q), cd_, this));
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
							holovibes_.get_chart_queue(),
							formatted_path,
							cd_.nb_frames_record,
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

			if (cd_.current_window == WindowKind::XYview)
				q = holovibes_.get_gpu_output_queue().get();
			else if (cd_.current_window == WindowKind::XZview)
				q = holovibes_.get_pipe()->get_stft_slice_queue(0).get();
			else
				q = holovibes_.get_pipe()->get_stft_slice_queue(1).get();

			std::string output_filename = format_batch_output(path, file_index_);
			output_filename = set_record_filename_properties(q->get_fd(), output_filename, false);
			cd_.nb_frames_record = frame_nb_spin_box->value();
			if (is_batch_img_)
			{
				try
				{
					if (gpib_interface_->execute_next_block())
					{
						record_thread_.reset(new ThreadRecorder(*q, output_filename, holo_file_get_json_settings(q), cd_, this));
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
							holovibes_.get_chart_queue(),
							output_filename,
							cd_.nb_frames_record,
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

			if (no_error)
				display_info("Batch record done");

			if (plot_window_)
			{
				plot_window_->stop_drawing();
				holovibes_.get_pipe()->request_chart(&holovibes_.get_chart_queue());
				plot_window_->start_drawing();
			}
		}

		void MainWindow::stop_csv_record()
		{
			if (cd_.chart_enabled)
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
		void MainWindow::set_start_stop_buttons(bool value)
		{
			ui.ImportStartPushButton->setEnabled(value);
			ui.ImportStopPushButton->setEnabled(value);
		}

		void MainWindow::import_browse_file()
		{
			static QString tmp_path = "";
			QString filename = "";

			filename = QFileDialog::getOpenFileName(this,
				tr("import file"), ((tmp_path == "") ? ("C://") : (tmp_path)), tr("All files (*.holo *.cine);; Holo files (*.holo);; Cine files (*.cine)"));

			QLineEdit* import_line_edit = ui.ImportPathLineEdit;
			import_line_edit->clear();
			import_line_edit->insert(filename);

			if (filename.endsWith(".holo"))
			{
				HoloFile* holo_file = HoloFile::new_instance(filename.toStdString());
				cd_.file_type = FileType::HOLO;

				set_start_stop_buttons(holo_file != nullptr);
			}

			else if (filename.endsWith(".cine"))
			{
				CineFile* cine_file = CineFile::new_instance(filename.toStdString());
				cd_.file_type = FileType::CINE;

				set_start_stop_buttons(cine_file != nullptr);
			}
		}

		void MainWindow::import_stop()
		{
			close_windows();
			close_critical_compute();
			camera_none();
			remove_infos();
			cd_.compute_mode = Computation::Stop;
			notify();
		}

		void MainWindow::import_start()
		{
			import_stop();

			if (cd_.file_type == FileType::HOLO)
				holo_file_update_cd();

			init_holovibes_import_mode();

			bool is_raw = is_raw_mode();
			ui.DirectRadioButton->setChecked(is_raw);
			ui.HologramRadioButton->setChecked(!is_raw);
		}

		void MainWindow::init_holovibes_import_mode()
		{
			QLineEdit *import_line_edit = ui.ImportPathLineEdit;
			QSpinBox *fps_spinbox = ui.ImportInputFpsSpinBox;
			QSpinBox *start_spinbox = ui.ImportStartIndexSpinBox;
			QCheckBox *load_file_gpu = ui.LoadFileInGpuCheckBox;
			QSpinBox *end_spinbox = ui.ImportEndIndexSpinBox;

			cd_.time_filter_stride = std::ceil(static_cast<float>(fps_spinbox->value()) / 20.0f);
			cd_.batch_size = cd_.time_filter_stride;

			FrameDescriptor fd;

			if (cd_.file_type == FileType::CINE)
				fd = get_cine_file_frame_descriptor();
			else
				fd = get_holo_file_frame_descriptor();

			is_enabled_camera_ = false;
			try
			{
				std::string file_src = import_line_edit->text().toUtf8();

				// TODO wrong calcul because header not taken into consideration
				// Will be useless with the new file system
				auto file_end = std::filesystem::file_size(file_src)
					/ fd.frame_size();
				if (file_end > end_spinbox->value())
					file_end = end_spinbox->value();

				set_correct_square_input_mode();

				holovibes_.init_import_mode(
					file_src,
					fd,
					true,
					fps_spinbox->value(),
					start_spinbox->value(),
					file_end,
					load_file_gpu->isChecked(),
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

		FrameDescriptor MainWindow::get_holo_file_frame_descriptor()
		{
			HoloFile* holo_file = HoloFile::get_instance();
			const HoloFile::Header& header = holo_file->get_header();

			cd_.pixel_size = holo_file->get_meta_data().value("pixel_size", 1.0f);

			width = header.img_width;
			height = header.img_height;
			get_good_size(width, height, window_max_size);

			FrameDescriptor fd = {
				static_cast<ushort>(header.img_width),
				static_cast<ushort>(header.img_height),
				header.pixel_bits / 8,
				header.endianess ? Endianness::BigEndian : Endianness::LittleEndian };

			return fd;
		}

		FrameDescriptor MainWindow::get_cine_file_frame_descriptor()
		{
			CineFile* cine_file = CineFile::get_instance();
			const CineFile::ImageInfo& image_info = cine_file->get_image_info();

			cd_.pixel_size = image_info.pixel_size;

			width = image_info.img_width;
			height = image_info.img_height;
			get_good_size(width, height, window_max_size);

			FrameDescriptor fd = {
				static_cast<ushort>(image_info.img_width),
				static_cast<ushort>(image_info.img_height),
				image_info.pixel_bits / 8,
				Endianness::LittleEndian };

			return fd;
		}

		void MainWindow::import_start_spinbox_update()
		{
			QSpinBox *start_spinbox = ui.ImportStartIndexSpinBox;
			QSpinBox *end_spinbox = ui.ImportEndIndexSpinBox;

			if (start_spinbox->value() > end_spinbox->value())
				end_spinbox->setValue(start_spinbox->value());
		}

		void MainWindow::import_end_spinbox_update()
		{
			QSpinBox *start_spinbox = ui.ImportStartIndexSpinBox;
			QSpinBox *end_spinbox = ui.ImportEndIndexSpinBox;

			if (end_spinbox->value() < start_spinbox->value())
				start_spinbox->setValue(end_spinbox->value());
		}

		void MainWindow::holo_file_update_cd()
		{
			auto holo_file = HoloFile::get_instance();
			const json& json_settings = holo_file->get_meta_data();

			cd_.compute_mode = json_settings.value("mode", Computation::Raw);
			cd_.algorithm = json_settings.value("algorithm", Algorithm::None);
			cd_.time_filter = json_settings.value("time_filter", TimeFilter::STFT);
			cd_.time_filter_size = json_settings.value("#img", 1);
			cd_.pindex = json_settings.value("p", 0);
			cd_.lambda = json_settings.value("lambda", 0.0f);
			cd_.pixel_size = json_settings.value("pixel_size", 12.0);
			cd_.zdistance = json_settings.value("z", 0.0f);
			cd_.log_scale_slice_xy_enabled = json_settings.value("log_scale", false);
			cd_.contrast_min_slice_xy = json_settings.value("contrast_min", 0.0f);
			cd_.contrast_max_slice_xy = json_settings.value("contrast_max", 0.0f);
			cd_.fft_shift_enabled = json_settings.value("fft_shift_enabled", true);
			cd_.x_accu_enabled = json_settings.value("x_acc_enabled", false);
			cd_.x_acc_level = json_settings.value("x_acc_level", 1);
			cd_.y_accu_enabled = json_settings.value("y_acc_enabled", false);
			cd_.y_acc_level = json_settings.value("y_acc_level", 1);
			cd_.p_accu_enabled = json_settings.value("p_acc_enabled", false);
			cd_.p_acc_level = json_settings.value("p_acc_level", 1);
			cd_.img_acc_slice_xy_enabled = json_settings.value("img_acc_slice_xy_enabled", false);
			cd_.img_acc_slice_xz_enabled = json_settings.value("img_acc_slice_xz_enabled", false);
			cd_.img_acc_slice_yz_enabled = json_settings.value("img_acc_slice_yz_enabled", false);
			cd_.img_acc_slice_xy_level = json_settings.value("img_acc_slice_xy_level", 1);
			cd_.img_acc_slice_xz_level = json_settings.value("img_acc_slice_xz_level", 1);
			cd_.img_acc_slice_yz_level = json_settings.value("img_acc_slice_yz_level", 1);
			cd_.renorm_enabled = json_settings.value("renorm_enabled", true);
			cd_.renorm_constant = json_settings.value("renorm_constant", 15);
		}

		json MainWindow::holo_file_get_json_settings(const Queue* q)
		{
			try
			{
				json json_settings;
				if (q != nullptr)
				{
					json_settings = HoloFile::get_json_settings(cd_, q->get_fd());
				}
				else
				{
					// This code shouldn't run but it's here to avoid a segfault in case something weird happens
					json_settings = HoloFile::get_json_settings(cd_);
					json_settings.emplace("img_width", window_max_size);
					json_settings.emplace("img_height", window_max_size);
					json_settings.emplace("pixel_bits", 16);
				}
				return json_settings;
			}
			catch (const std::exception& e)
			{
				LOG_ERROR(e.what());
				return json();
			}
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

		RawWindow *MainWindow::get_main_display()
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
