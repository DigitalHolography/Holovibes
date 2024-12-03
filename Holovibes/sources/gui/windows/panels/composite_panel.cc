/*! \file
 *
 */

#include <filesystem>

#include "composite_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "API.hh"
#include "GUI.hh"

namespace api = ::holovibes::api;

namespace holovibes::gui
{
CompositePanel::CompositePanel(QWidget* parent)
    : Panel(parent)
{
    hide();
}

CompositePanel::~CompositePanel() {}

void CompositePanel::showEvent(QShowEvent* event)
{
    const unsigned min_val_composite = api::get_time_transformation_size() == 1 ? 0 : 1;
    const unsigned max_val_composite = api::get_time_transformation_size() - 1;

    ui_->PRedSpinBox_Composite->setValue(min_val_composite);
    ui_->SpinBox_hue_freq_min->setValue(min_val_composite);
    ui_->SpinBox_saturation_freq_min->setValue(min_val_composite);
    ui_->SpinBox_value_freq_min->setValue(min_val_composite);

    ui_->PBlueSpinBox_Composite->setValue(max_val_composite);
    ui_->SpinBox_hue_freq_max->setValue(max_val_composite);
    ui_->SpinBox_saturation_freq_max->setValue(max_val_composite);
    ui_->SpinBox_value_freq_max->setValue(max_val_composite);
}

void CompositePanel::on_notify()
{
    if (!isVisible())
        return;

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

    // RGB
    QSpinBoxQuietSetValue(ui_->PRedSpinBox_Composite, api::get_composite_p_red());
    QSpinBoxQuietSetValue(ui_->PBlueSpinBox_Composite, api::get_composite_p_blue());
    ui_->WeightSpinBox_R->setValue(api::get_weight_r());
    ui_->WeightSpinBox_G->setValue(api::get_weight_g());
    ui_->WeightSpinBox_B->setValue(api::get_weight_b());
    // -- RGB

    // HSV
    ui_->CompositePanel->actualize_frequency_channel_v();

    QSpinBoxQuietSetValue(ui_->SpinBox_hue_freq_min, api::get_composite_p_min_h());
    QSpinBoxQuietSetValue(ui_->SpinBox_hue_freq_max, api::get_composite_p_max_h());
    QSliderQuietSetValue(ui_->horizontalSlider_hue_threshold_min,
                         static_cast<int>(api::get_slider_h_threshold_min() * 1000));
    ui_->CompositePanel->slide_update_threshold_h_min();
    QSliderQuietSetValue(ui_->horizontalSlider_hue_threshold_max,
                         static_cast<int>(api::get_slider_h_threshold_max() * 1000));
    ui_->CompositePanel->slide_update_threshold_h_max();
    QSliderQuietSetValue(ui_->horizontalSlider_hue_shift_min, static_cast<int>(api::get_slider_h_shift_min() * 1000));
    ui_->CompositePanel->slide_update_shift_h_min();
    QSliderQuietSetValue(ui_->horizontalSlider_hue_shift_max, static_cast<int>(api::get_slider_h_shift_max() * 1000));
    ui_->CompositePanel->slide_update_shift_h_max();

    QSpinBoxQuietSetValue(ui_->SpinBox_saturation_freq_min, api::get_composite_p_min_s());
    QSpinBoxQuietSetValue(ui_->SpinBox_saturation_freq_max, api::get_composite_p_max_s());
    QSliderQuietSetValue(ui_->horizontalSlider_saturation_threshold_min,
                         static_cast<int>(api::get_slider_s_threshold_min() * 1000));
    ui_->CompositePanel->slide_update_threshold_s_min();
    QSliderQuietSetValue(ui_->horizontalSlider_saturation_threshold_max,
                         static_cast<int>(api::get_slider_s_threshold_max() * 1000));
    ui_->CompositePanel->slide_update_threshold_s_max();

    QSpinBoxQuietSetValue(ui_->SpinBox_value_freq_min, api::get_composite_p_min_v());
    QSpinBoxQuietSetValue(ui_->SpinBox_value_freq_max, api::get_composite_p_max_v());
    QSliderQuietSetValue(ui_->horizontalSlider_value_threshold_min,
                         static_cast<int>(api::get_slider_v_threshold_min() * 1000));
    ui_->CompositePanel->slide_update_threshold_v_min();
    QSliderQuietSetValue(ui_->horizontalSlider_value_threshold_max,
                         static_cast<int>(api::get_slider_v_threshold_max() * 1000));
    ui_->CompositePanel->slide_update_threshold_v_max();
    // -- HSV

    bool rgbMode = (api::get_composite_kind() == CompositeKind::RGB);

    ui_->radioButton_rgb->setChecked(rgbMode);
    ui_->radioButton_hsv->setChecked(!rgbMode);

    ui_->groupBox->setVisible(rgbMode);   // Frequency channel
    ui_->groupBox_5->setVisible(rgbMode); // Color equalization box
    ui_->RenormalizationCheckBox->setVisible(rgbMode);
    ui_->CompositeAreaButton->setVisible(rgbMode);

    ui_->groupBox_hue->setVisible(!rgbMode);
    ui_->groupBox_saturation->setVisible(!rgbMode);
    ui_->groupBox_value->setVisible(!rgbMode);

    ui_->zFFTShiftCheckBox->setChecked(api::get_z_fft_shift());
}

void CompositePanel::click_z_fft_shift(bool checked)
{
    api::set_z_fft_shift(checked);
    parent_->notify();
}

void CompositePanel::set_composite_intervals()
{
    // PRedSpinBox_Composite value cannont be higher than PBlueSpinBox_Composite
    ui_->PRedSpinBox_Composite->setValue(
        std::min(ui_->PRedSpinBox_Composite->value(), ui_->PBlueSpinBox_Composite->value()));

    api::set_rgb_p(ui_->PRedSpinBox_Composite->value(), ui_->PBlueSpinBox_Composite->value());

    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_h_min()
{
    api::set_composite_p_min_h(ui_->SpinBox_hue_freq_min->value());
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_h_max()
{
    api::set_composite_p_max_h(ui_->SpinBox_hue_freq_max->value());
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_s_min()
{
    api::set_composite_p_min_s(ui_->SpinBox_saturation_freq_min->value());
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_s_max()
{
    api::set_composite_p_max_s(ui_->SpinBox_saturation_freq_max->value());
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_v_min()
{
    api::set_composite_p_min_v(ui_->SpinBox_value_freq_min->value());
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_v_max()
{
    api::set_composite_p_max_v(ui_->SpinBox_value_freq_max->value());
    parent_->notify();
}

void CompositePanel::set_composite_weights()
{
    api::set_weight_rgb(ui_->WeightSpinBox_R->value(), ui_->WeightSpinBox_G->value(), ui_->WeightSpinBox_B->value());
    parent_->notify();
}

void CompositePanel::set_composite_auto_weights(bool value)
{
    api::set_composite_auto_weights(value);

    ui_->WeightSpinBox_R->setEnabled(!value);
    ui_->WeightSpinBox_G->setEnabled(!value);
    ui_->WeightSpinBox_B->setEnabled(!value);

    parent_->notify();
}

void CompositePanel::click_composite_rgb_or_hsv()
{
    if (ui_->radioButton_rgb->isChecked())
    {
        api::set_composite_kind(CompositeKind::RGB);
        ui_->PRedSpinBox_Composite->setValue(ui_->SpinBox_hue_freq_min->value());
        ui_->PBlueSpinBox_Composite->setValue(ui_->SpinBox_hue_freq_max->value());
    }
    else
    {
        api::set_composite_kind(CompositeKind::HSV);
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
    receiver = slider.value() / 1000.0f;

    if (lower_bound > upper_bound)
    {
        bound_to_update = slider.value() / 1000.0f;
        slider_to_update.setValue(slider.value());
    }

    char array[10];
    sprintf_s(array, "%d", slider.value());
    fancy_Qslide_text_percent(array);
    to_be_written_in.setText(QString(array));
}

void CompositePanel::slide_update_threshold_h_min()
{

    // Avoid modification from panel instead of API
    float receiver = api::get_slider_h_threshold_min();
    float bound_to_update = api::get_slider_h_threshold_max();

    slide_update_threshold(*ui_->horizontalSlider_hue_threshold_min,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_hue_threshold_max,
                           *ui_->label_hue_threshold_min,
                           api::get_slider_h_threshold_min(),
                           api::get_slider_h_threshold_max());

    api::set_slider_h_threshold_min(receiver);
    api::set_slider_h_threshold_max(bound_to_update);
}

void CompositePanel::slide_update_threshold_h_max()
{

    float receiver = api::get_slider_h_threshold_max();
    float bound_to_update = api::get_slider_h_threshold_min();

    slide_update_threshold(*ui_->horizontalSlider_hue_threshold_max,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_hue_threshold_min,
                           *ui_->label_hue_threshold_max,
                           api::get_slider_h_threshold_min(),
                           api::get_slider_h_threshold_max());

    api::set_slider_h_threshold_max(receiver);
    api::set_slider_h_threshold_min(bound_to_update);
}

void CompositePanel::slide_update_shift_h_min()
{
    // Avoid modification from panel instead of API
    float receiver = api::get_slider_h_shift_min();
    float bound_to_update = api::get_slider_h_shift_max();

    slide_update_threshold(*ui_->horizontalSlider_hue_shift_min,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_hue_shift_max,
                           *ui_->label_hue_shift_min,
                           api::get_slider_h_shift_min(),
                           api::get_slider_h_shift_max());

    api::set_slider_h_shift_min(receiver);
    api::set_slider_h_shift_max(bound_to_update);
}

void CompositePanel::slide_update_shift_h_max()
{

    float receiver = api::get_slider_h_shift_max();
    float bound_to_update = api::get_slider_h_shift_min();

    slide_update_threshold(*ui_->horizontalSlider_hue_shift_max,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_hue_shift_min,
                           *ui_->label_hue_shift_max,
                           api::get_slider_h_shift_min(),
                           api::get_slider_h_shift_max());

    api::set_slider_h_shift_max(receiver);
    api::set_slider_h_shift_min(bound_to_update);
}

void CompositePanel::slide_update_threshold_s_min()
{

    float receiver = api::get_slider_s_threshold_min();
    float bound_to_update = api::get_slider_s_threshold_max();

    slide_update_threshold(*ui_->horizontalSlider_saturation_threshold_min,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_saturation_threshold_max,
                           *ui_->label_saturation_threshold_min,
                           api::get_slider_s_threshold_min(),
                           api::get_slider_s_threshold_max());

    api::set_slider_s_threshold_min(receiver);
    api::set_slider_s_threshold_max(bound_to_update);
}

void CompositePanel::slide_update_threshold_s_max()
{

    float receiver = api::get_slider_s_threshold_max();
    float bound_to_update = api::get_slider_s_threshold_min();

    slide_update_threshold(*ui_->horizontalSlider_saturation_threshold_max,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_saturation_threshold_min,
                           *ui_->label_saturation_threshold_max,
                           api::get_slider_s_threshold_min(),
                           api::get_slider_s_threshold_max());

    api::set_slider_s_threshold_max(receiver);
    api::set_slider_s_threshold_min(bound_to_update);
}

void CompositePanel::slide_update_threshold_v_min()
{

    float receiver = api::get_slider_v_threshold_min();
    float bound_to_update = api::get_slider_v_threshold_max();

    slide_update_threshold(*ui_->horizontalSlider_value_threshold_min,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_value_threshold_max,
                           *ui_->label_value_threshold_min,
                           api::get_slider_v_threshold_min(),
                           api::get_slider_v_threshold_max());

    api::set_slider_v_threshold_min(receiver);
    api::set_slider_v_threshold_max(bound_to_update);
}

void CompositePanel::slide_update_threshold_v_max()
{

    float receiver = api::get_slider_v_threshold_max();
    float bound_to_update = api::get_slider_v_threshold_min();

    slide_update_threshold(*ui_->horizontalSlider_value_threshold_max,
                           receiver,
                           bound_to_update,
                           *ui_->horizontalSlider_value_threshold_min,
                           *ui_->label_value_threshold_max,
                           api::get_slider_v_threshold_min(),
                           api::get_slider_v_threshold_max());

    api::set_slider_v_threshold_max(receiver);
    api::set_slider_v_threshold_min(bound_to_update);
}

void CompositePanel::actualize_frequency_channel_s()
{
    api::set_composite_p_activated_s(ui_->checkBox_saturation_freq->isChecked());

    ui_->SpinBox_saturation_freq_min->setDisabled(!ui_->checkBox_saturation_freq->isChecked());
    ui_->SpinBox_saturation_freq_max->setDisabled(!ui_->checkBox_saturation_freq->isChecked());
}

void CompositePanel::actualize_frequency_channel_v()
{
    api::set_composite_p_activated_v(ui_->checkBox_value_freq->isChecked());

    ui_->SpinBox_value_freq_min->setDisabled(!ui_->checkBox_value_freq->isChecked());
    ui_->SpinBox_value_freq_max->setDisabled(!ui_->checkBox_value_freq->isChecked());
}

void CompositePanel::set_composite_area() { gui::set_composite_area(); }
} // namespace holovibes::gui
