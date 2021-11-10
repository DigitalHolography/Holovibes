#include "asw_panel_advanced.hh"

namespace holovibes::gui
{
ASWPanelAdvanced::ASWPanelAdvanced(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "Advanced")
{
    advanced_layout_ = new QVBoxLayout();

    // File spin box
    display_rate_ = new QDoubleSpinBoxLayout(parent, parent_widget, "DisplayRate");
    display_rate_->setValue(0);
    advanced_layout_->addItem(display_rate_);
    connect(display_rate_, SIGNAL(value_changed()), this, SLOT(on_change_display_rate_value()));

    // Input spin box
    filter2d_smooth_low_ = new QDoubleSpinBoxLayout(parent, parent_widget, "Filter2D_smooth_low");
    filter2d_smooth_low_->setValue(0);
    advanced_layout_->addItem(filter2d_smooth_low_);
    connect(filter2d_smooth_low_, SIGNAL(value_changed()), this, SLOT(on_change_filter2d_smooth_low_value()));

    // Input spin box
    filter2d_smooth_high_ = new QDoubleSpinBoxLayout(parent, parent_widget, "Filter2D_smooth_high");
    filter2d_smooth_high_->setValue(0.5f);
    advanced_layout_->addItem(filter2d_smooth_high_);
    connect(filter2d_smooth_high_, SIGNAL(value_changed()), this, SLOT(on_change_filter2d_smooth_high_value()));

    // Record spin box
    contrast_upper_threshold_ = new QDoubleSpinBoxLayout(parent, parent_widget, "Contrast_upper_threshold");
    contrast_upper_threshold_->setValue(99.5f);
    advanced_layout_->addItem(contrast_upper_threshold_);
    connect(contrast_upper_threshold_, SIGNAL(value_changed()), this, SLOT(on_change_contrast_upper_threshold_value()));

    // Output spin box
    renorm_constant_ = new QIntSpinBoxLayout(parent, parent_widget, "Renorm_constant");
    renorm_constant_->setValue(5);
    advanced_layout_->addItem(renorm_constant_);
    connect(renorm_constant_, SIGNAL(value_changed()), this, SLOT(on_change_renorm_constant_value()));

    // 3D cuts spin box
    cuts_contrast_p_offset_ = new QIntSpinBoxLayout(parent, parent_widget, "Cuts_contrast_p_offset");
    cuts_contrast_p_offset_->setValue(0);
    advanced_layout_->addItem(cuts_contrast_p_offset_);
    connect(cuts_contrast_p_offset_, SIGNAL(value_changed()), this, SLOT(on_change_cuts_contrast_p_offset_value()));

    setLayout(advanced_layout_);
}

ASWPanelAdvanced::~ASWPanelAdvanced() {}

#pragma region SLOTS
// TODO: region to implement with API
void ASWPanelAdvanced::on_change_display_rate_value() { LOG_INFO << display_rate_->get_value(); };

void ASWPanelAdvanced::on_change_filter2d_smooth_low_value() { LOG_INFO << filter2d_smooth_low_->get_value(); };

void ASWPanelAdvanced::on_change_filter2d_smooth_high_value() { LOG_INFO << filter2d_smooth_high_->get_value(); };

void ASWPanelAdvanced::on_change_contrast_upper_threshold_value()
{
    LOG_INFO << contrast_upper_threshold_->get_value();
};

void ASWPanelAdvanced::on_change_renorm_constant_value() { LOG_INFO << renorm_constant_->get_value(); };

void ASWPanelAdvanced::on_change_cuts_contrast_p_offset_value() { LOG_INFO << cuts_contrast_p_offset_->get_value(); };
#pragma endregion

} // namespace holovibes::gui