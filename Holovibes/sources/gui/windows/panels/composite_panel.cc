/*! \file
 *
 */

#include <filesystem>

#include "composite_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "API.hh"

namespace api = ::holovibes::api;

namespace holovibes::gui
{
CompositePanel::CompositePanel(QWidget* parent)
    : Panel(parent)
{
    hide();
}

CompositePanel::~CompositePanel() {}

void CompositePanel::on_notify()
{

    const int time_transformation_size_max = api::get_time_transformation_size() - 1;
    ui_->PRedSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui_->PBlueSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui_->SpinBox_hue_freq_min->setMaximum(time_transformation_size_max);
    ui_->SpinBox_hue_freq_max->setMaximum(time_transformation_size_max);
    ui_->SpinBox_saturation_freq_min->setMaximum(time_transformation_size_max);
    ui_->SpinBox_saturation_freq_max->setMaximum(time_transformation_size_max);
    ui_->SpinBox_value_freq_min->setMaximum(time_transformation_size_max);
    ui_->SpinBox_value_freq_max->setMaximum(time_transformation_size_max);

    ui_->RenormalizationCheckBox->setChecked(api::get_composite_auto_weights());

    QSpinBoxQuietSetValue(ui_->PRedSpinBox_Composite, api::get_composite_rgb().get_red());
    QSpinBoxQuietSetValue(ui_->PBlueSpinBox_Composite, api::get_composite_rgb().get_blue());
    QDoubleSpinBoxQuietSetValue(ui_->WeightSpinBox_R, api::get_composite_rgb().weight.r);
    QDoubleSpinBoxQuietSetValue(ui_->WeightSpinBox_G, api::get_composite_rgb().weight.g);
    QDoubleSpinBoxQuietSetValue(ui_->WeightSpinBox_B, api::get_composite_rgb().weight.b);
    ui_->CompositePanel->actualize_frequency_channel_v();

    QSpinBoxQuietSetValue(ui_->SpinBox_hue_freq_min, api::get_composite_hsv().h.p.min);
    QSpinBoxQuietSetValue(ui_->SpinBox_hue_freq_max, api::get_composite_hsv().h.p.max);
    QSliderQuietSetValue(ui_->horizontalSlider_hue_threshold_min,
                         static_cast<int>(api::get_composite_hsv().h.threshold.min * 1000));
    ui_->CompositePanel->slide_update_threshold_h_min();
    QSliderQuietSetValue(ui_->horizontalSlider_hue_threshold_max,
                         static_cast<int>(api::get_composite_hsv().h.threshold.max * 1000));
    ui_->CompositePanel->slide_update_threshold_h_max();

    QSpinBoxQuietSetValue(ui_->SpinBox_saturation_freq_min, api::get_composite_hsv().s.threshold.min);
    QSpinBoxQuietSetValue(ui_->SpinBox_saturation_freq_max, api::get_composite_hsv().s.threshold.max);
    QSliderQuietSetValue(ui_->horizontalSlider_saturation_threshold_min,
                         static_cast<int>(api::get_composite_hsv().s.threshold.min * 1000));
    ui_->CompositePanel->slide_update_threshold_s_min();
    QSliderQuietSetValue(ui_->horizontalSlider_saturation_threshold_max,
                         static_cast<int>(api::get_composite_hsv().s.threshold.max * 1000));
    ui_->CompositePanel->slide_update_threshold_s_max();

    QSpinBoxQuietSetValue(ui_->SpinBox_value_freq_min, api::get_composite_hsv().v.p.min);
    QSpinBoxQuietSetValue(ui_->SpinBox_value_freq_max, api::get_composite_hsv().v.p.max);
    QSliderQuietSetValue(ui_->horizontalSlider_value_threshold_min,
                         static_cast<int>(api::get_composite_hsv().v.threshold.min * 1000));
    ui_->CompositePanel->slide_update_threshold_v_min();
    QSliderQuietSetValue(ui_->horizontalSlider_value_threshold_max,
                         static_cast<int>(api::get_composite_hsv().v.threshold.max * 1000));
    ui_->CompositePanel->slide_update_threshold_v_max();

    bool rgbMode = ui_->radioButton_rgb->isChecked();

    auto show_rgb = [this, rgbMode]()
    {
        ui_->groupBox->setVisible(rgbMode);
        ui_->groupBox_5->setVisible(rgbMode || ui_->RenormalizationCheckBox->isChecked());
    };

    auto show_hsv = [this, rgbMode]()
    {
        ui_->groupBox_hue->setHidden(rgbMode);
        ui_->groupBox_saturation->setHidden(rgbMode);
        ui_->groupBox_value->setHidden(rgbMode);
    };

    if (rgbMode)
    {
        show_hsv();
        show_rgb();
    }
    else
    {
        show_rgb();
        show_hsv();
    }
}

void CompositePanel::set_composite_intervals()
{
    // PRedSpinBox_Composite value cannont be higher than PBlueSpinBox_Composite
    ui_->PRedSpinBox_Composite->setValue(
        std::min(ui_->PRedSpinBox_Composite->value(), ui_->PBlueSpinBox_Composite->value()));

    api::change_composite_rgb()->set_red(ui_->PRedSpinBox_Composite->value());
    api::change_composite_rgb()->set_blue(ui_->PBlueSpinBox_Composite->value());
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_h_min()
{
    api::change_composite_hsv()->h.p.min = ui_->SpinBox_hue_freq_min->value();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_h_max()
{
    api::change_composite_hsv()->h.p.max = ui_->SpinBox_hue_freq_max->value();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_s_min()
{
    api::change_composite_hsv()->s.p.min = ui_->SpinBox_saturation_freq_min->value();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_s_max()
{
    api::change_composite_hsv()->s.p.max = ui_->SpinBox_saturation_freq_max->value();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_v_min()
{
    api::change_composite_hsv()->v.p.min = ui_->SpinBox_value_freq_min->value();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_v_max()
{
    api::change_composite_hsv()->v.p.max = ui_->SpinBox_value_freq_max->value();
    parent_->notify();
}

void CompositePanel::set_composite_weights()
{
    api::change_composite_rgb()->weight.r = (ui_->WeightSpinBox_R->value());
    api::change_composite_rgb()->weight.g = (ui_->WeightSpinBox_G->value());
    api::change_composite_rgb()->weight.b = (ui_->WeightSpinBox_B->value());

    parent_->notify();
}

void CompositePanel::set_composite_auto_weights(bool value)
{
    api::set_composite_auto_weights(value);
    ui_->ViewPanel->set_auto_contrast();
}

void CompositePanel::click_composite_rgb_or_hsv()
{
    if (ui_->radioButton_rgb->isChecked())
    {
        api::set_composite_kind(CompositeKindEnum::RGB);
        ui_->PRedSpinBox_Composite->setValue(ui_->SpinBox_hue_freq_min->value());
        ui_->PBlueSpinBox_Composite->setValue(ui_->SpinBox_hue_freq_max->value());
    }
    else
    {
        api::set_composite_kind(CompositeKindEnum::HSV);
        ui_->SpinBox_hue_freq_min->setValue(ui_->PRedSpinBox_Composite->value());
        ui_->SpinBox_hue_freq_max->setValue(ui_->PBlueSpinBox_Composite->value());
        ui_->SpinBox_saturation_freq_min->setValue(ui_->PRedSpinBox_Composite->value());
        ui_->SpinBox_saturation_freq_max->setValue(ui_->PBlueSpinBox_Composite->value());
        ui_->SpinBox_value_freq_min->setValue(ui_->PRedSpinBox_Composite->value());
        ui_->SpinBox_value_freq_max->setValue(ui_->PBlueSpinBox_Composite->value());
    }

    parent_->notify();
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

void slide_update_threshold(const QSlider& slider,
                            float& receiver,
                            float& bound_to_update,
                            QSlider& slider_to_update,
                            QLabel& to_be_written_in,
                            const float lower_bound,
                            const float& upper_bound)
{

    const bool res = api::slide_update_threshold(slider.value(), receiver, bound_to_update, lower_bound, upper_bound);

    char array[10];
    sprintf_s(array, "%d", slider.value());
    fancy_Qslide_text_percent(array);
    to_be_written_in.setText(QString(array));

    if (res)
        slider_to_update.setValue(slider.value());
}

void CompositePanel::slide_update_threshold_h_min()
{

    // Avoid modification from panel instead of API
    float receiver = api::get_composite_hsv().h.threshold.min;
    float bound_to_update = api::get_composite_hsv().h.threshold.max;

    slide_update_threshold(*ui_->horizontalSlider_hue_threshold_min,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_hue_threshold_max,
                           *ui_->label_hue_threshold_min,
                           api::get_composite_hsv().h.threshold.min,
                           api::get_composite_hsv().h.threshold.max);

    api::change_composite_hsv()->h.threshold.min = receiver;
    api::change_composite_hsv()->h.threshold.max = bound_to_update;
}

void CompositePanel::slide_update_threshold_h_max()
{

    float receiver = api::get_composite_hsv().h.threshold.max;
    float bound_to_update = api::get_composite_hsv().h.threshold.min;

    slide_update_threshold(*ui_->horizontalSlider_hue_threshold_max,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_hue_threshold_min,
                           *ui_->label_hue_threshold_max,
                           api::get_composite_hsv().h.threshold.min,
                           api::get_composite_hsv().h.threshold.max);

    api::change_composite_hsv()->h.threshold.max = receiver;
    api::change_composite_hsv()->h.threshold.min = bound_to_update;
}

void CompositePanel::slide_update_threshold_s_min()
{

    float receiver = api::get_composite_hsv().s.threshold.min;
    float bound_to_update = api::get_composite_hsv().s.threshold.max;

    slide_update_threshold(*ui_->horizontalSlider_saturation_threshold_min,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_saturation_threshold_max,
                           *ui_->label_saturation_threshold_min,
                           api::get_composite_hsv().s.threshold.min,
                           api::get_composite_hsv().s.threshold.max);

    api::change_composite_hsv()->s.threshold.min = receiver;
    api::change_composite_hsv()->s.threshold.max = bound_to_update;
}

void CompositePanel::slide_update_threshold_s_max()
{

    float receiver = api::get_composite_hsv().s.threshold.max;
    float bound_to_update = api::get_composite_hsv().s.threshold.min;

    slide_update_threshold(*ui_->horizontalSlider_saturation_threshold_max,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_saturation_threshold_min,
                           *ui_->label_saturation_threshold_max,
                           api::get_composite_hsv().s.threshold.min,
                           api::get_composite_hsv().s.threshold.max);

    api::change_composite_hsv()->s.threshold.max = receiver;
    api::change_composite_hsv()->s.threshold.min = bound_to_update;
}

void CompositePanel::slide_update_threshold_v_min()
{

    float receiver = api::get_composite_hsv().v.threshold.min;
    float bound_to_update = api::get_composite_hsv().v.threshold.max;

    slide_update_threshold(*ui_->horizontalSlider_value_threshold_min,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_value_threshold_max,
                           *ui_->label_value_threshold_min,
                           api::get_composite_hsv().v.threshold.min,
                           api::get_composite_hsv().v.threshold.max);

    api::change_composite_hsv()->v.threshold.min = receiver;
    api::change_composite_hsv()->v.threshold.max = bound_to_update;
}

void CompositePanel::slide_update_threshold_v_max()
{

    float receiver = api::get_composite_hsv().v.threshold.max;
    float bound_to_update = api::get_composite_hsv().v.threshold.min;

    slide_update_threshold(*ui_->horizontalSlider_value_threshold_max,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_value_threshold_min,
                           *ui_->label_value_threshold_max,
                           api::get_composite_hsv().v.threshold.min,
                           api::get_composite_hsv().v.threshold.max);

    api::change_composite_hsv()->v.threshold.max = receiver;
    api::change_composite_hsv()->v.threshold.min = bound_to_update;
}

void CompositePanel::actualize_frequency_channel_s()
{
    api::change_composite_hsv()->s.p.activated = ui_->checkBox_saturation_freq->isChecked();

    ui_->SpinBox_saturation_freq_min->setDisabled(!ui_->checkBox_saturation_freq->isChecked());
    ui_->SpinBox_saturation_freq_max->setDisabled(!ui_->checkBox_saturation_freq->isChecked());
}

void CompositePanel::actualize_frequency_channel_v()
{
    api::change_composite_hsv()->v.p.activated = ui_->checkBox_value_freq->isChecked();

    ui_->SpinBox_value_freq_min->setDisabled(!ui_->checkBox_value_freq->isChecked());
    ui_->SpinBox_value_freq_max->setDisabled(!ui_->checkBox_value_freq->isChecked());
}

void CompositePanel::actualize_checkbox_h_gaussian_blur()
{
    api::change_composite_hsv()->h.blur.enabled = ui_->checkBox_h_gaussian_blur->isChecked();

    ui_->SpinBox_hue_blur_kernel_size->setEnabled(ui_->checkBox_h_gaussian_blur->isChecked());
}

void CompositePanel::actualize_kernel_size_blur()
{
    api::change_composite_hsv()->h.blur.kernel_size = ui_->SpinBox_hue_blur_kernel_size->value();
}

void CompositePanel::set_composite_area() { api::set_composite_area(); }
} // namespace holovibes::gui
