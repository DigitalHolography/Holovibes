#include "view_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"

#include "view_struct.hh"

#include "API.hh"

#define MIN_IMG_NB_TIME_TRANSFORMATION_CUTS 8

namespace api = ::holovibes::api;

namespace holovibes::gui
{
ViewPanel::ViewPanel(QWidget* parent)
    : Panel(parent)
{
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

void ViewPanel::on_notify()
{
    const bool is_raw = api::is_raw_mode();

    ui_->ViewModeComboBox->setCurrentIndex(static_cast<int>(api::get_img_type()));

    ui_->PhaseUnwrap2DCheckBox->setEnabled(api::get_img_type() == ImgType::PhaseIncrease ||
                                           api::get_img_type() == ImgType::Argument);

    ui_->TimeTransformationCutsCheckBox->setChecked(!is_raw && api::get_time_transformation_cuts_enabled());
    ui_->TimeTransformationCutsCheckBox->setEnabled(ui_->timeTransformationSizeSpinBox->value() >=
                                                    MIN_IMG_NB_TIME_TRANSFORMATION_CUTS);

    ui_->FFTShiftCheckBox->setChecked(api::get_fft_shift_enabled());
    ui_->FFTShiftCheckBox->setEnabled(true);

    ui_->LensViewCheckBox->setChecked(api::get_lens_view_enabled());

    ui_->RawDisplayingCheckBox->setEnabled(!is_raw);
    ui_->RawDisplayingCheckBox->setChecked(!is_raw && api::get_raw_view_enabled());

    // Contrast
    ui_->ContrastCheckBox->setChecked(!is_raw && api::get_contrast_enabled());
    ui_->ContrastCheckBox->setEnabled(true);
    ui_->AutoRefreshContrastCheckBox->setChecked(api::get_contrast_auto_refresh());
    ui_->InvertContrastCheckBox->setChecked(api::get_contrast_invert_enabled());
    ui_->ContrastMinDoubleSpinBox->setEnabled(!api::get_contrast_auto_refresh());
    ui_->ContrastMinDoubleSpinBox->setValue(api::get_contrast_min());
    ui_->ContrastMaxDoubleSpinBox->setEnabled(!api::get_contrast_auto_refresh());
    ui_->ContrastMaxDoubleSpinBox->setValue(api::get_contrast_max());

    // Window selection
    QComboBox* window_selection = ui_->WindowSelectionComboBox;
    window_selection->setEnabled(!is_raw);
    window_selection->setCurrentIndex(static_cast<int>(api::get_current_window()));

    // Log
    ui_->LogScaleCheckBox->setEnabled(true);
    ui_->LogScaleCheckBox->setChecked(!is_raw && api::get_img_log_scale_slice_enabled());

    // ImgAccWindow
    auto set_xyzf_visibility = [&](bool val) {
        ui_->ImgAccuLabel->setVisible(val);
        ui_->ImgAccuSpinBox->setVisible(val);
        ui_->RotatePushButton->setVisible(val);
        ui_->FlipPushButton->setVisible(val);
    };

    if (api::get_current_window() == WindowKind::Filter2D)
        set_xyzf_visibility(false);
    else
    {
        set_xyzf_visibility(true);

        ui_->ImgAccuSpinBox->setValue(api::get_img_accu_level());

        ui_->RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(api::get_rotation()))).c_str());
        ui_->FlipPushButton->setText(("Flip " + std::to_string(api::get_flip_enabled())).c_str());
    }

    // p accu
    ui_->PAccSpinBox->setMaximum(api::get_time_transformation_size() - 1);

    api::get_cd().check_p_limits(); // FIXME: May be moved in setters
    ui_->PAccSpinBox->setValue(api::get_p_accu_level());
    ui_->PSpinBox->setValue(api::get_pindex());
    ui_->PAccSpinBox->setEnabled(api::get_img_type() != ImgType::PhaseIncrease);

    ui_->PSpinBox->setMaximum(api::get_time_transformation_size() - api::get_p_accu_level() - 1);
    ui_->PAccSpinBox->setMaximum(api::get_time_transformation_size() - api::get_pindex() - 1);
    ui_->PSpinBox->setEnabled(!is_raw);

    // q accu
    bool is_ssa_stft = api::get_time_transformation() == TimeTransformation::SSA_STFT;
    ui_->Q_AccSpinBox->setEnabled(is_ssa_stft && !is_raw);
    ui_->Q_SpinBox->setEnabled(is_ssa_stft && !is_raw);

    ui_->Q_AccSpinBox->setMaximum(api::get_time_transformation_size() - 1);

    api::get_cd().check_q_limits(); // FIXME: May be moved in setters
    ui_->Q_AccSpinBox->setValue(api::get_q_accu_level());
    ui_->Q_SpinBox->setValue(api::get_q_index());
    ui_->Q_SpinBox->setMaximum(api::get_time_transformation_size() - api::get_q_accu_level() - 1);
    ui_->Q_AccSpinBox->setMaximum(api::get_time_transformation_size() - api::get_q_index() - 1);

    // XY accu
    ui_->XAccSpinBox->setValue(api::get_x_accu_level());
    ui_->YAccSpinBox->setValue(api::get_y_accu_level());

    int max_width = 0;
    int max_height = 0;
    if (api::get_gpu_input_queue() != nullptr)
    {
        max_width = api::get_gpu_input_queue_fd_width() - 1;
        max_height = api::get_gpu_input_queue_fd_height() - 1;
    }
    else
    {
        api::set_x_y(0, 0);
    }

    ui_->XSpinBox->setMaximum(max_width);
    ui_->YSpinBox->setMaximum(max_height);
    QSpinBoxQuietSetValue(ui_->XSpinBox, api::get_x_cuts());
    QSpinBoxQuietSetValue(ui_->YSpinBox, api::get_y_cuts());

    ui_->RenormalizeCheckBox->setChecked(api::get_renorm_enabled());
    ui_->ReticleScaleDoubleSpinBox->setEnabled(api::get_reticle_display_enabled());
    ui_->ReticleScaleDoubleSpinBox->setValue(api::get_reticle_scale());
    ui_->DisplayReticleCheckBox->setChecked(api::get_reticle_display_enabled());
}

void ViewPanel::load_gui(const boost::property_tree::ptree& ptree)
{
    bool h = ptree.get<bool>("window.view_hidden", isHidden());
    ui_->actionView->setChecked(!h);
    setHidden(h);

    time_transformation_cuts_window_max_size =
        ptree.get<uint>("window_size.time_transformation_cuts_window_max_size", 512);
}

void ViewPanel::save_gui(boost::property_tree::ptree& ptree)
{
    ptree.put<bool>("window.view_hidden", isHidden());
    ptree.put<uint>("window_size.time_transformation_cuts_window_max_size", time_transformation_cuts_window_max_size);
}

void ViewPanel::set_view_mode(const QString& value) { parent_->set_view_image_type(value); }

void ViewPanel::set_unwrapping_2d(const bool value)
{
    if (api::is_raw_mode())
        return;

    api::set_unwrapping_2d(value);

    parent_->notify();
}

void ViewPanel::toggle_time_transformation_cuts(bool checked)
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    QComboBox* winSelection = ui_->WindowSelectionComboBox;
    winSelection->setEnabled(checked);
    winSelection->setCurrentIndex((!checked) ? 0 : winSelection->currentIndex());

    if (!checked)
    {
        cancel_time_transformation_cuts();
        return;
    }

    const ushort nImg = api::get_time_transformation_size();
    uint time_transformation_size = std::max(256u, std::min(512u, (uint)nImg));

    if (time_transformation_size > time_transformation_cuts_window_max_size)
        time_transformation_size = time_transformation_cuts_window_max_size;

    const bool res = api::toggle_time_transformation_cuts(time_transformation_size);

    if (res)
    {
        set_auto_contrast_cuts();
        parent_->notify();
    }
    else
    {
        cancel_time_transformation_cuts();
    }
}

void ViewPanel::cancel_time_transformation_cuts()
{
    if (!api::get_time_transformation_cuts_enabled())
        return;

    std::function<void()> callback = []() { return; };

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
    {
        callback = ([=]() {
            api::set_time_transformation_cuts_enabled(false);
            pipe->delete_stft_slice_queue();

            ui_->TimeTransformationCutsCheckBox->setChecked(false);
            parent_->notify();
        });
    }

    api::cancel_time_transformation_cuts(callback);

    parent_->notify();
}

void ViewPanel::set_auto_contrast_cuts() { api::set_auto_contrast_cuts(); }

void ViewPanel::set_fft_shift(const bool value)
{
    if (api::is_raw_mode())
        return;

    api::set_fft_shift(value);

    api::pipe_refresh();
}

void ViewPanel::update_lens_view(bool value)
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    api::set_lens_view_enabled(value);

    if (value)
    {
        const bool res = api::set_lens_view(parent_->auxiliary_window_max_size);

        if (res)
        {
            connect(UserInterfaceDescriptor::instance().lens_window.get(),
                    SIGNAL(destroyed()),
                    this,
                    SLOT(disable_lens_view()));
        }
    }
    else
    {
        disable_lens_view();
    }

    api::pipe_refresh();
}

void ViewPanel::disable_lens_view()
{
    if (UserInterfaceDescriptor::instance().lens_window)
        disconnect(UserInterfaceDescriptor::instance().lens_window.get(),
                   SIGNAL(destroyed()),
                   this,
                   SLOT(disable_lens_view()));

    api::disable_lens_view();
    parent_->notify();
}

void ViewPanel::update_raw_view(bool value)
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    if (value)
    {
        if (api::get_batch_size() > api::get_output_buffer_size())
        {
            LOG_ERROR << "[RAW VIEW] Batch size must be lower than output queue size";
            return;
        }

        api::set_raw_view(parent_->auxiliary_window_max_size);
        connect(api::get_raw_window().get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));
    }
    else
    {
        disable_raw_view();
    }
}
void ViewPanel::disable_raw_view()
{
    if (UserInterfaceDescriptor::instance().raw_window)
        disconnect(UserInterfaceDescriptor::instance().raw_window.get(),
                   SIGNAL(destroyed()),
                   this,
                   SLOT(disable_raw_view()));

    api::disable_raw_view();

    parent_->notify();
}

void ViewPanel::set_x_y() { api::set_x_y(ui_->XSpinBox->value(), ui_->YSpinBox->value()); }

void ViewPanel::set_x_accu()
{
    api::set_x_accu(ui_->XAccSpinBox->value());

    parent_->notify();
}

void ViewPanel::set_y_accu()
{
    api::set_y_accu(ui_->YAccSpinBox->value());

    parent_->notify();
}

void ViewPanel::set_p(int value)
{
    if (api::is_raw_mode())
        return;

    if (value >= static_cast<int>(api::get_time_transformation_size()))
    {
        LOG_ERROR << "p param has to be between 1 and #img";
        return;
    }

    api::set_p(value);

    parent_->notify();
}

void ViewPanel::increment_p()
{
    if (api::is_raw_mode())
        return;

    // FIXME: Cannot append
    if (api::get_pindex() >= api::get_time_transformation_size())
    {
        LOG_ERROR << "p param has to be between 1 and #img";
        return;
    }

    set_p(api::get_pindex() + 1);
    set_auto_contrast();

    parent_->notify();
}

void ViewPanel::decrement_p()
{
    if (api::is_raw_mode())
        return;

    // FIXME: Cannot append
    if (api::get_pindex() <= 0)
    {
        LOG_ERROR << "p param has to be between 1 and #img";
        return;
    }

    set_p(api::get_pindex() - 1);
    set_auto_contrast();

    parent_->notify();
}

void ViewPanel::set_p_accu()
{
    api::set_p_accu(ui_->PAccSpinBox->value());

    parent_->notify();
}

void ViewPanel::set_q(int value)
{
    api::set_q(value);

    parent_->notify();
}

void ViewPanel::set_q_acc()
{
    api::set_q_accu(ui_->Q_AccSpinBox->value());

    parent_->notify();
}

void ViewPanel::rotateTexture()
{
    api::rotateTexture();

    parent_->notify();
}

void ViewPanel::flipTexture()
{
    api::flipTexture();

    parent_->notify();
}

void ViewPanel::set_log_scale(const bool value)
{
    if (api::is_raw_mode())
        return;

    api::set_log_scale(value);

    parent_->notify();
}

void ViewPanel::set_accumulation_level(int value)
{
    if (api::is_raw_mode())
        return;

    api::set_accumulation_level(value);
}

void ViewPanel::set_contrast_mode(bool value)
{
    if (api::is_raw_mode())
        return;

    api::set_contrast_mode(value);

    parent_->notify();
}

void ViewPanel::set_auto_contrast()
{
    if (api::is_raw_mode())
        return;

    api::set_auto_contrast();
}

void ViewPanel::set_auto_refresh_contrast(bool value)
{
    api::set_auto_refresh_contrast(value);

    parent_->notify();
}

void ViewPanel::invert_contrast(bool value)
{
    if (api::is_raw_mode())
        return;

    if (!api::get_contrast_enabled())
        return;

    api::invert_contrast(value);
}

void ViewPanel::set_contrast_min(const double value)
{
    if (api::is_raw_mode())
        return;

    if (!api::get_contrast_enabled())
        return;

    api::set_contrast_min(value);
}

void ViewPanel::set_contrast_max(const double value)
{
    if (api::is_raw_mode())
        return;

    if (!api::get_contrast_enabled())
        return;

    api::set_contrast_max(value);
}

void ViewPanel::toggle_renormalize(bool value) { api::toggle_renormalize(value); }

void ViewPanel::display_reticle(bool value)
{
    api::display_reticle(value);

    parent_->notify();
}

void ViewPanel::reticle_scale(double value)
{
    if (!is_between(value, 0., 1.))
        return;

    api::reticle_scale(value);
}
} // namespace holovibes::gui
