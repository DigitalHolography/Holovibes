#include <filesystem>

#include "composite_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"

namespace holovibes::gui
{
CompositePanel::CompositePanel(QWidget* parent)
    : Panel(parent)
{
}

CompositePanel::~CompositePanel() {}

void CompositePanel::on_notify()
{
    const int time_transformation_size_max = parent_->cd_.time_transformation_size - 1;
    ui_->PRedSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui_->PBlueSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui_->SpinBox_hue_freq_min->setMaximum(time_transformation_size_max);
    ui_->SpinBox_hue_freq_max->setMaximum(time_transformation_size_max);
    ui_->SpinBox_saturation_freq_min->setMaximum(time_transformation_size_max);
    ui_->SpinBox_saturation_freq_max->setMaximum(time_transformation_size_max);
    ui_->SpinBox_value_freq_min->setMaximum(time_transformation_size_max);
    ui_->SpinBox_value_freq_max->setMaximum(time_transformation_size_max);

    ui_->RenormalizationCheckBox->setChecked(parent_->cd_.composite_auto_weights_);

    parent_->QSpinBoxQuietSetValue(ui_->PRedSpinBox_Composite, parent_->cd_.composite_p_red);
    parent_->QSpinBoxQuietSetValue(ui_->PBlueSpinBox_Composite, parent_->cd_.composite_p_blue);
    parent_->QDoubleSpinBoxQuietSetValue(ui_->WeightSpinBox_R, parent_->cd_.weight_r);
    parent_->QDoubleSpinBoxQuietSetValue(ui_->WeightSpinBox_G, parent_->cd_.weight_g);
    parent_->QDoubleSpinBoxQuietSetValue(ui_->WeightSpinBox_B, parent_->cd_.weight_b);
    ui_->CompositePanel->actualize_frequency_channel_v();

    parent_->QSpinBoxQuietSetValue(ui_->SpinBox_hue_freq_min, parent_->cd_.composite_p_min_h);
    parent_->QSpinBoxQuietSetValue(ui_->SpinBox_hue_freq_max, parent_->cd_.composite_p_max_h);
    parent_->QSliderQuietSetValue(ui_->horizontalSlider_hue_threshold_min,
                                  (int)(parent_->cd_.slider_h_threshold_min * 1000));
    ui_->CompositePanel->slide_update_threshold_h_min();
    parent_->QSliderQuietSetValue(ui_->horizontalSlider_hue_threshold_max,
                                  (int)(parent_->cd_.slider_h_threshold_max * 1000));
    ui_->CompositePanel->slide_update_threshold_h_max();

    parent_->QSpinBoxQuietSetValue(ui_->SpinBox_saturation_freq_min, parent_->cd_.composite_p_min_s);
    parent_->QSpinBoxQuietSetValue(ui_->SpinBox_saturation_freq_max, parent_->cd_.composite_p_max_s);
    parent_->QSliderQuietSetValue(ui_->horizontalSlider_saturation_threshold_min,
                                  (int)(parent_->cd_.slider_s_threshold_min * 1000));
    ui_->CompositePanel->slide_update_threshold_s_min();
    parent_->QSliderQuietSetValue(ui_->horizontalSlider_saturation_threshold_max,
                                  (int)(parent_->cd_.slider_s_threshold_max * 1000));
    ui_->CompositePanel->slide_update_threshold_s_max();

    parent_->QSpinBoxQuietSetValue(ui_->SpinBox_value_freq_min, parent_->cd_.composite_p_min_v);
    parent_->QSpinBoxQuietSetValue(ui_->SpinBox_value_freq_max, parent_->cd_.composite_p_max_v);
    parent_->QSliderQuietSetValue(ui_->horizontalSlider_value_threshold_min,
                                  (int)(parent_->cd_.slider_v_threshold_min * 1000));
    ui_->CompositePanel->slide_update_threshold_v_min();
    parent_->QSliderQuietSetValue(ui_->horizontalSlider_value_threshold_max,
                                  (int)(parent_->cd_.slider_v_threshold_max * 1000));
    ui_->CompositePanel->slide_update_threshold_v_max();

    bool rgbMode = ui_->radioButton_rgb->isChecked();
    ui_->groupBox->setHidden(!rgbMode);
    ui_->groupBox_5->setHidden(!rgbMode && !ui_->RenormalizationCheckBox->isChecked());
    ui_->groupBox_hue->setHidden(rgbMode);
    ui_->groupBox_saturation->setHidden(rgbMode);
    ui_->groupBox_value->setHidden(rgbMode);
}

void CompositePanel::set_composite_intervals()
{
    // PRedSpinBox_Composite value cannont be higher than PBlueSpinBox_Composite
    ui_->PRedSpinBox_Composite->setValue(
        std::min(ui_->PRedSpinBox_Composite->value(), ui_->PBlueSpinBox_Composite->value()));
    parent_->cd_.set_composite_p_red(ui_->PRedSpinBox_Composite->value());
    parent_->cd_.set_composite_p_blue(ui_->PBlueSpinBox_Composite->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_h_min()
{
    parent_->cd_.set_composite_p_min_h(ui_->SpinBox_hue_freq_min->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_h_max()
{
    parent_->cd_.set_composite_p_max_h(ui_->SpinBox_hue_freq_max->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_s_min()
{
    parent_->cd_.set_composite_p_min_s(ui_->SpinBox_saturation_freq_min->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_s_max()
{
    parent_->cd_.set_composite_p_max_s(ui_->SpinBox_saturation_freq_max->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_v_min()
{
    parent_->cd_.set_composite_p_min_v(ui_->SpinBox_value_freq_min->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void CompositePanel::set_composite_intervals_hsv_v_max()
{
    parent_->cd_.set_composite_p_max_v(ui_->SpinBox_value_freq_max->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void CompositePanel::set_composite_weights()
{
    parent_->cd_.set_weight_rgb(ui_->WeightSpinBox_R->value(),
                                ui_->WeightSpinBox_G->value(),
                                ui_->WeightSpinBox_B->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void CompositePanel::set_composite_auto_weights(bool value)
{
    parent_->cd_.set_composite_auto_weights(value);
    ui_->ViewPanel->set_auto_contrast();
}

void CompositePanel::click_composite_rgb_or_hsv()
{
    parent_->cd_.set_composite_kind(ui_->radioButton_rgb->isChecked() ? CompositeKind::RGB : CompositeKind::HSV);
    if (ui_->radioButton_rgb->isChecked())
    {
        ui_->PRedSpinBox_Composite->setValue(ui_->SpinBox_hue_freq_min->value());
        ui_->PBlueSpinBox_Composite->setValue(ui_->SpinBox_hue_freq_max->value());
    }
    else
    {
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

void slide_update_threshold(QSlider& slider,
                            std::atomic<float>& receiver,
                            std::atomic<float>& bound_to_update,
                            QSlider& slider_to_update,
                            QLabel& to_be_written_in,
                            std::atomic<float>& lower_bound,
                            std::atomic<float>& upper_bound)
{
    // Store the slider value in parent_->cd_ (ComputeDescriptor)
    receiver = slider.value() / 1000.0f;

    char array[10];
    sprintf_s(array, "%d", slider.value());
    fancy_Qslide_text_percent(array);
    to_be_written_in.setText(QString(array));

    if (lower_bound > upper_bound)
    {
        // FIXME bound_to_update = receiver ?
        bound_to_update = slider.value() / 1000.0f;

        slider_to_update.setValue(slider.value());
    }
}

void CompositePanel::slide_update_threshold_h_min()
{
    slide_update_threshold(*ui_->horizontalSlider_hue_threshold_min,
                           parent_->cd_.slider_h_threshold_min,
                           parent_->cd_.slider_h_threshold_max,
                           *ui_->horizontalSlider_hue_threshold_max,
                           *ui_->label_hue_threshold_min,
                           parent_->cd_.slider_h_threshold_min,
                           parent_->cd_.slider_h_threshold_max);
}

void CompositePanel::slide_update_threshold_h_max()
{
    slide_update_threshold(*ui_->horizontalSlider_hue_threshold_max,
                           parent_->cd_.slider_h_threshold_max,
                           parent_->cd_.slider_h_threshold_min,
                           *ui_->horizontalSlider_hue_threshold_min,
                           *ui_->label_hue_threshold_max,
                           parent_->cd_.slider_h_threshold_min,
                           parent_->cd_.slider_h_threshold_max);
}

void CompositePanel::slide_update_threshold_s_min()
{
    slide_update_threshold(*ui_->horizontalSlider_saturation_threshold_min,
                           parent_->cd_.slider_s_threshold_min,
                           parent_->cd_.slider_s_threshold_max,
                           *ui_->horizontalSlider_saturation_threshold_max,
                           *ui_->label_saturation_threshold_min,
                           parent_->cd_.slider_s_threshold_min,
                           parent_->cd_.slider_s_threshold_max);
}

void CompositePanel::slide_update_threshold_s_max()
{
    slide_update_threshold(*ui_->horizontalSlider_saturation_threshold_max,
                           parent_->cd_.slider_s_threshold_max,
                           parent_->cd_.slider_s_threshold_min,
                           *ui_->horizontalSlider_saturation_threshold_min,
                           *ui_->label_saturation_threshold_max,
                           parent_->cd_.slider_s_threshold_min,
                           parent_->cd_.slider_s_threshold_max);
}

void CompositePanel::slide_update_threshold_v_min()
{
    slide_update_threshold(*ui_->horizontalSlider_value_threshold_min,
                           parent_->cd_.slider_v_threshold_min,
                           parent_->cd_.slider_v_threshold_max,
                           *ui_->horizontalSlider_value_threshold_max,
                           *ui_->label_value_threshold_min,
                           parent_->cd_.slider_v_threshold_min,
                           parent_->cd_.slider_v_threshold_max);
}

void CompositePanel::slide_update_threshold_v_max()
{
    slide_update_threshold(*ui_->horizontalSlider_value_threshold_max,
                           parent_->cd_.slider_v_threshold_max,
                           parent_->cd_.slider_v_threshold_min,
                           *ui_->horizontalSlider_value_threshold_min,
                           *ui_->label_value_threshold_max,
                           parent_->cd_.slider_v_threshold_min,
                           parent_->cd_.slider_v_threshold_max);
}

void CompositePanel::actualize_frequency_channel_s()
{
    parent_->cd_.set_composite_p_activated_s(ui_->checkBox_saturation_freq->isChecked());
    ui_->SpinBox_saturation_freq_min->setDisabled(!ui_->checkBox_saturation_freq->isChecked());
    ui_->SpinBox_saturation_freq_max->setDisabled(!ui_->checkBox_saturation_freq->isChecked());
}

void CompositePanel::actualize_frequency_channel_v()
{
    parent_->cd_.set_composite_p_activated_v(ui_->checkBox_value_freq->isChecked());
    ui_->SpinBox_value_freq_min->setDisabled(!ui_->checkBox_value_freq->isChecked());
    ui_->SpinBox_value_freq_max->setDisabled(!ui_->checkBox_value_freq->isChecked());
}

void CompositePanel::actualize_checkbox_h_gaussian_blur()
{
    parent_->cd_.set_h_blur_activated(ui_->checkBox_h_gaussian_blur->isChecked());
    ui_->SpinBox_hue_blur_kernel_size->setEnabled(ui_->checkBox_h_gaussian_blur->isChecked());
}

void CompositePanel::actualize_kernel_size_blur()
{
    parent_->cd_.set_h_blur_kernel_size(ui_->SpinBox_hue_blur_kernel_size->value());
}

void CompositePanel::set_composite_area() { parent_->mainDisplay->getOverlayManager().create_overlay<CompositeArea>(); }
} // namespace holovibes::gui
