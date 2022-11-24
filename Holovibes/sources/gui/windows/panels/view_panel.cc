/*! \file
 *
 */

#include <limits>

#include "view_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"

#include "view_struct.hh"

#include "API.hh"
#include "gui_utilities.hh"

#define MIN_IMG_NB_TIME_TRANSFORMATION_CUTS 8

namespace api = ::holovibes::api;

namespace holovibes::gui
{
ViewPanel::ViewPanel(QWidget* parent)
    : Panel(parent)
{
    UserInterface::instance().view_panel = this;
    p_left_shortcut_ = new QShortcut(QKeySequence("Left"), this);
    p_left_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_left_shortcut_, SIGNAL(activated()), this, SLOT(decrement_p()));

    p_right_shortcut_ = new QShortcut(QKeySequence("Right"), this);
    p_right_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_right_shortcut_, SIGNAL(activated()), this, SLOT(increment_p()));
}

ViewPanel::~ViewPanel()
{
    delete p_left_shortcut_;
    delete p_right_shortcut_;
}

// TODO: use parameters instead of directly the GSH
void ViewPanel::view_callback(WindowKind, ViewWindow)
{
    const bool is_raw = api::get_compute_mode() == Computation::Raw;

    ui_->ContrastCheckBox->setChecked(!is_raw && api::get_current_view().contrast.enabled);
    ui_->ContrastCheckBox->setEnabled(true);
    ui_->AutoRefreshContrastCheckBox->setChecked(api::get_current_view().contrast.auto_refresh);
    ui_->InvertContrastCheckBox->setChecked(api::get_current_view().contrast.invert);
    ui_->ContrastMinDoubleSpinBox->setEnabled(!api::get_current_view().contrast.auto_refresh);
    ui_->ContrastMinDoubleSpinBox->setValue(api::get_contrast_min());
    ui_->ContrastMaxDoubleSpinBox->setEnabled(!api::get_current_view().contrast.auto_refresh);
    ui_->ContrastMaxDoubleSpinBox->setValue(api::get_contrast_max());

    // Window selection
    QComboBox* window_selection = ui_->WindowSelectionComboBox;
    window_selection->setEnabled(!is_raw);
    window_selection->setCurrentIndex(static_cast<int>(api::get_current_view_kind()));
}

void ViewPanel::on_notify()
{
    const bool is_raw = api::get_compute_mode() == Computation::Raw;

    ui_->ViewModeComboBox->setCurrentIndex(static_cast<int>(api::get_image_type()));

    ui_->PhaseUnwrap2DCheckBox->setVisible(api::get_image_type() == ImageTypeEnum::PhaseIncrease ||
                                           api::get_image_type() == ImageTypeEnum::Argument);

    ui_->TimeTransformationCutsCheckBox->setEnabled(ui_->timeTransformationSizeSpinBox->value() >=
                                                    MIN_IMG_NB_TIME_TRANSFORMATION_CUTS);
    ui_->TimeTransformationCutsCheckBox->setChecked(!is_raw && api::get_cuts_view_enabled());

    ui_->FFTShiftCheckBox->setChecked(api::get_fft_shift_enabled());
    ui_->FFTShiftCheckBox->setEnabled(true);

    ui_->LensViewCheckBox->setChecked(api::get_lens_view_enabled());

    ui_->RawDisplayingCheckBox->setEnabled(!is_raw);
    ui_->RawDisplayingCheckBox->setChecked(!is_raw && api::get_raw_view_enabled());

    // Contrast
    ui_->ContrastCheckBox->setChecked(!is_raw && api::get_current_view().contrast.enabled);
    ui_->ContrastCheckBox->setEnabled(true);
    ui_->AutoRefreshContrastCheckBox->setChecked(api::get_current_view().contrast.auto_refresh);
    ui_->InvertContrastCheckBox->setChecked(api::get_current_view().contrast.invert);
    ui_->ContrastMinDoubleSpinBox->setEnabled(!api::get_current_view().contrast.auto_refresh);
    ui_->ContrastMinDoubleSpinBox->setValue(api::get_contrast_min());
    ui_->ContrastMaxDoubleSpinBox->setEnabled(!api::get_current_view().contrast.auto_refresh);
    ui_->ContrastMaxDoubleSpinBox->setValue(api::get_contrast_max());

    // Window selection
    QComboBox* window_selection = ui_->WindowSelectionComboBox;
    window_selection->setEnabled(!is_raw);
    window_selection->setCurrentIndex(static_cast<int>(api::get_current_view_kind()));

    // Log
    ui_->LogScaleCheckBox->setEnabled(true);
    ui_->LogScaleCheckBox->setChecked(!is_raw && api::get_current_view().log_enabled);

    // ImgAccWindow
    auto set_xyzf_visibility = [&](bool val)
    {
        ui_->ImgAccuLabel->setVisible(val);
        ui_->ImgAccuSpinBox->setVisible(val);
        ui_->RotatePushButton->setVisible(val);
        ui_->FlipPushButton->setVisible(val);
    };

    if (api::get_current_view_kind() == WindowKind::ViewFilter2D)
        set_xyzf_visibility(false);
    else
    {
        set_xyzf_visibility(true);

        ui_->ImgAccuSpinBox->setValue(api::get_current_view_as_view_xyz().img_accu_level);

        ui_->RotatePushButton->setText(
            ("Rot " + std::to_string(static_cast<int>(api::get_current_view_as_view_xyz().rot))).c_str());
        ui_->FlipPushButton->setText(
            ("Flip " + std::to_string(api::get_current_view_as_view_xyz().flip_enabled)).c_str());
    }

    // Deactivate previous maximum (chetor)
    ui_->PSpinBox->setMaximum(INT_MAX);
    ui_->PAccSpinBox->setMaximum(INT_MAX);

    // p accu

    ui_->PAccSpinBox->setValue(api::get_view_accu_p().accu_level);
    ui_->PSpinBox->setValue(api::get_view_accu_p().index);
    ui_->PAccSpinBox->setEnabled(api::get_image_type() != ImageTypeEnum::PhaseIncrease);

    api::check_p_limits(); // FIXME: May be moved in setters

    // Enforce maximum value for p_index and p_accu_level
    ui_->PSpinBox->setMaximum(api::get_time_transformation_size() - api::get_view_accu_p().accu_level - 1);
    ui_->PAccSpinBox->setMaximum(api::get_time_transformation_size() - api::get_view_accu_p().index - 1);
    ui_->PSpinBox->setEnabled(!is_raw);

    // q accu
    bool is_ssa_stft = api::get_time_transformation() == TimeTransformationEnum::SSA_STFT;
    ui_->Q_AccSpinBox->setVisible(is_ssa_stft && !is_raw);
    ui_->Q_SpinBox->setVisible(is_ssa_stft && !is_raw);
    ui_->Q_Label->setVisible(is_ssa_stft && !is_raw);
    ui_->QaccLabel->setVisible(is_ssa_stft && !is_raw);

    // Deactivate previous maximum (chetor + unused)
    ui_->Q_SpinBox->setMaximum(INT_MAX);
    ui_->Q_AccSpinBox->setMaximum(INT_MAX);

    ui_->Q_AccSpinBox->setValue(api::get_view_accu_q().accu_level);
    ui_->Q_SpinBox->setValue(api::get_view_accu_q().index);

    api::check_q_limits(); // FIXME: May be moved in setters
    ui_->Q_SpinBox->setMaximum(api::get_time_transformation_size() - api::get_view_accu_q().accu_level - 1);
    ui_->Q_AccSpinBox->setMaximum(api::get_time_transformation_size() - api::get_view_accu_q().index - 1);

    // XY accu
    ui_->XAccSpinBox->setValue(api::get_view_accu_x().accu_level);
    ui_->YAccSpinBox->setValue(api::get_view_accu_y().accu_level);

    int max_width = 0;
    int max_height = 0;
    if (api::get_gpu_input_queue_ptr() != nullptr)
    {
        max_width = api::get_gpu_input_queue().get_fd().width - 1;
        max_height = api::get_gpu_input_queue().get_fd().height - 1;
    }
    else
    {
        api::change_view_accu_x()->cuts = 0;
        api::change_view_accu_y()->cuts = 0;
    }

    ui_->XSpinBox->setMaximum(max_width);
    ui_->YSpinBox->setMaximum(max_height);
    QSpinBoxQuietSetValue(ui_->XSpinBox, api::get_view_accu_x().cuts);
    QSpinBoxQuietSetValue(ui_->YSpinBox, api::get_view_accu_y().cuts);

    ui_->RenormalizeCheckBox->setChecked(api::get_renorm_enabled());
    ui_->ReticleScaleDoubleSpinBox->setEnabled(api::get_reticle().display_enabled);
    ui_->ReticleScaleDoubleSpinBox->setValue(api::get_reticle().reticle_scale);
    ui_->DisplayReticleCheckBox->setChecked(api::get_reticle().display_enabled);
}

void ViewPanel::load_gui(const json& j_us)
{
    bool h = json_get_or_default(j_us, isHidden(), "panels", "view hidden", isHidden());
    ui_->actionView->setChecked(!h);
    setHidden(h);

    time_transformation_cuts_window_max_size =
        json_get_or_default(j_us, 512, "windows", "time transformation cuts window max size");
}

void ViewPanel::save_gui(json& j_us)
{
    j_us["panels"]["view hidden"] = isHidden();
    j_us["windows"]["time transformation cuts window max size"] = time_transformation_cuts_window_max_size;
}

void ViewPanel::set_view_mode(const QString& value) { parent_->set_view_image_type(value); }

void ViewPanel::set_unwrapping_2d(const bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;
    GSH::instance().set_value<Unwrap2DRequested>(value);

    parent_->notify();
}

void ViewPanel::update_3d_cuts_view(bool checked)
{
    if (api::get_import_type() == ImportTypeEnum::None)
        return;

    if (checked)
    {
        const ushort nImg = api::get_time_transformation_size();
        uint time_transformation_size = std::max(256u, std::min(512u, (uint)nImg));

        if (time_transformation_size > time_transformation_cuts_window_max_size)
            time_transformation_size = time_transformation_cuts_window_max_size;

        const bool res = api::set_3d_cuts_view(time_transformation_size);

        if (res)
        {
            set_auto_contrast_cuts();
            parent_->notify();
        }
        else
            cancel_time_transformation_cuts();
    }
    // FIXME: if slice are closed, cancel time should be call.
    else
        cancel_time_transformation_cuts();
}

void ViewPanel::cancel_time_transformation_cuts()
{
    if (!api::get_cuts_view_enabled())
        return;

    std::function<void()> callback = ([=]() {
        api::detail::set_value<TimeTransformationCutsEnable>(false);
        parent_->notify();
    });

    api::cancel_time_transformation_cuts(callback);
}

void ViewPanel::set_auto_contrast_cuts()
{
    api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXZ);
    api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewYZ);
}

void ViewPanel::set_fft_shift(const bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    api::set_fft_shift_enabled(value);
}

void set_raw_view(bool checked)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    api::detail::set_value<RawViewEnabled>(checked);

    GSH::instance().set_value<RawViewEnabled>(checked);
    while (api::get_compute_pipe().get_view_cache().has_change_requested())
        continue;

    // FIXME API : Need to move this outside this (and this function must be useless)
    if (checked)
    {
        const FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
        ushort raw_window_width = fd.width;
        ushort raw_window_height = fd.height;
        get_good_size(raw_window_width, raw_window_height, UserInterface::auxiliary_window_max_size);

        // set positions of new windows according to the position of the main GL
        // window and Lens window
        QPoint pos = UserInterface::instance().main_display->framePosition() +
                     QPoint(UserInterface::instance().main_display->width() + 310, 0);
        UserInterface::instance().raw_window.reset(
            new gui::RawWindow(pos,
                               QSize(raw_window_width, raw_window_height),
                               api::get_compute_pipe().get_raw_view_queue_ptr().get()));

        UserInterface::instance().raw_window->setTitle("Raw view");
    }
    else
    {
        UserInterface::instance().raw_window.reset(nullptr);
    }
}

void set_lens_view(bool checked)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    api::set_lens_view_enabled(checked);
    while (api::get_compute_pipe().get_view_cache().has_change_requested())
        continue;

    // FIXME API : Need to move this outside this (and this function must be useless)
    if (checked)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = UserInterface::instance().main_display->framePosition() +
                         QPoint(UserInterface::instance().main_display->width() + 310, 0);

            const FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
            ushort lens_window_width = fd.width;
            ushort lens_window_height = fd.height;
            get_good_size(lens_window_width, lens_window_height, auxiliary_window_max_size);

            UserInterface::instance().lens_window.reset(
                new gui::RawWindow(pos,
                                   QSize(lens_window_width, lens_window_height),
                                   get_compute_pipe().get_fourier_transforms().get_lens_queue().get(),
                                   0.f,
                                   gui::KindOfView::Lens));

            UserInterface::instance().lens_window->setTitle("Lens view");
        }
        catch (const std::exception& e)
        {
            LOG_ERROR(main, "Catch {}", e.what());
        }
    }
    else
    {
        UserInterface::instance().lens_window.reset(nullptr);
    }
}

void ViewPanel::update_lens_view(bool checked)
{
    if (api::get_import_type() == ImportTypeEnum::None)
        return;

    api::set_lens_view(checked, parent_->auxiliary_window_max_size);
}

void ViewPanel::update_raw_view(bool checked)
{
    if (api::get_import_type() == ImportTypeEnum::None)
        return;

    if (checked && api::get_batch_size() > api::get_output_buffer_size())
    {
        LOG_ERROR("[RAW VIEW] Batch size must be lower than output queue size");
        return;
    }

    api::set_raw_view(checked, parent_->auxiliary_window_max_size);
}

void ViewPanel::close_windows()
{
    UserInterface::instance().main_display.reset(nullptr);

    UserInterface::instance().sliceXZ.reset(nullptr);
    UserInterface::instance().sliceYZ.reset(nullptr);
    UserInterface::instance().filter2d_window.reset(nullptr);

    if (UserInterface::instance().lens_window)
        set_lens_view(false, 0);
    if (UserInterface::instance().raw_window)
        set_raw_view(false, 0);

    UserInterface::instance().plot_window_.reset(nullptr);
}

void ViewPanel::set_x_y()
{
    api::set_x_cuts(ui_->XSpinBox->value());
    api::set_y_cuts(ui_->YSpinBox->value());
}

void ViewPanel::set_x_accu()
{
    api::change_view_accu_x()->accu_level = ui_->XAccSpinBox->value();

    parent_->notify();
}

void ViewPanel::set_y_accu()
{
    api::change_view_accu_y()->accu_level = ui_->YAccSpinBox->value();

    parent_->notify();
}

void ViewPanel::set_p(int value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    if (value >= static_cast<int>(api::get_time_transformation_size()))
    {
        LOG_ERROR("p param has to be between 1 and #img");
        return;
    }

    api::change_view_accu_p()->index = value;

    parent_->notify();
}

void ViewPanel::increment_p()
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    // FIXME: Cannot append
    if (api::get_view_accu_p().index >= api::get_time_transformation_size())
    {
        LOG_ERROR("p param has to be between 1 and #img");
        return;
    }

    set_p(api::get_view_accu_p().index + 1);
    parent_->notify();
}

void ViewPanel::decrement_p()
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    // FIXME: Cannot append
    if (api::get_view_accu_p().index <= 0)
    {
        LOG_ERROR("p param has to be between 1 and #img");
        return;
    }

    set_p(api::get_view_accu_p().index - 1);
    parent_->notify();
}

void ViewPanel::set_p_accu()
{
    api::change_view_accu_p()->accu_level = ui_->PAccSpinBox->value();

    parent_->notify();
}

void ViewPanel::set_q(int value)
{
    api::change_view_accu_q()->index = value;

    parent_->notify();
}

void ViewPanel::set_q_acc()
{
    api::change_view_accu_q()->accu_level = ui_->Q_AccSpinBox->value();

    parent_->notify();
}

void ViewPanel::rotateTexture()
{
    double rot = api::get_current_view_as_view_xyz().rot;
    double new_rot = (rot == 270.f) ? 0.f : rot + 90.f;
    api::change_current_view_as_view_xyz()->rot = new_rot;

    if (api::get_current_view_kind() == WindowKind::ViewXY)
        UserInterface::instance().main_display->setTransform();
    else if (UserInterface::instance().sliceXZ && api::get_current_view_kind() == WindowKind::ViewXZ)
        UserInterface::instance().sliceXZ->setTransform();
    else if (UserInterface::instance().sliceYZ && api::get_current_view_kind() == WindowKind::ViewYZ)
        UserInterface::instance().sliceYZ->setTransform();

    parent_->notify();
}

void ViewPanel::flipTexture()
{
    api::change_current_view_as_view_xyz()->flip_enabled = !api::get_current_view_as_view_xyz().flip_enabled;

    // FIXME => TRIGGER
    if (api::get_current_view_kind() == WindowKind::ViewXY)
        UserInterface::instance().main_display->setTransform();
    else if (UserInterface::instance().sliceXZ && api::get_current_view_kind() == WindowKind::ViewXZ)
        UserInterface::instance().sliceXZ->setTransform();
    else if (UserInterface::instance().sliceYZ && api::get_current_view_kind() == WindowKind::ViewYZ)
        UserInterface::instance().sliceYZ->setTransform();

    parent_->notify();
}

void ViewPanel::set_log_scale(const bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    api::set_log_scale(value);

    parent_->notify();
}

void ViewPanel::set_accumulation_level(int value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    api::change_current_view_as_view_xyz()->img_accu_level = value;
}

void ViewPanel::set_contrast_mode(bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    api::change_current_view()->contrast.enabled = value;

    parent_->notify();
}

void ViewPanel::set_auto_contrast()
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    api::request_exec_contrast_current_window();
}

void ViewPanel::set_auto_refresh_contrast(bool value)
{
    api::change_current_view()->contrast.auto_refresh = value;

    parent_->notify();
}

void ViewPanel::invert_contrast(bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    if (!api::get_current_view().contrast.enabled)
        return;

    api::change_current_view()->contrast.invert = value;
}

void ViewPanel::set_contrast_min(const double value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    if (!api::get_current_view().contrast.enabled)
        return;

    api::set_current_window_contrast_min(value);
}

void ViewPanel::set_contrast_max(const double value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    if (!api::get_current_view().contrast.enabled)
        return;

    api::set_current_window_contrast_max(value);
}

void ViewPanel::toggle_renormalize(bool value)
{
    set_renorm_enabled(value);

    if (api::get_import_type() != ImportTypeEnum::None)
    {
        api::get_compute_pipe().get_rendering().request_view_clear_image_accumulation(WindowKind::ViewXY);
        api::get_compute_pipe().get_rendering().request_view_clear_image_accumulation(WindowKind::ViewXZ);
        api::get_compute_pipe().get_rendering().request_view_clear_image_accumulation(WindowKind::ViewYZ);
    }
}

void ViewPanel::display_reticle(bool value)
{
    if (api::get_reticle().display_enabled != value)
        gui::utilities::display_reticle(value);

    parent_->notify();
}

void ViewPanel::reticle_scale(double value)
{
    if (!is_between(value, 0., 1.))
        return;

    api::change_reticle()->reticle_scale = value;
}
} // namespace holovibes::gui
