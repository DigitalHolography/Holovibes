#include "asw_panel_advanced.hh"

namespace holovibes::gui
{

ASWPanelAdvanced::ASWPanelAdvanced(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "Advanced")
{
    advanced_layout_ = new QVBoxLayout(parent);

    // File spin box
    display_rate_ = new QDoubleSpinBoxLayout(parent, parent_widget, "DisplayRate");
    display_rate_->setValue(0);
    advanced_layout_->addItem(display_rate_);

    // Input spin box
    filter2d_smooth_low_ = new QDoubleSpinBoxLayout(parent, parent_widget, "Filter2D_smooth_low");
    filter2d_smooth_low_->setValue(0);
    advanced_layout_->addItem(filter2d_smooth_low_);

    // Input spin box
    filter2d_smooth_high_ = new QDoubleSpinBoxLayout(parent, parent_widget, "Filter2D_smooth_high");
    filter2d_smooth_high_->setValue(0.5f);
    advanced_layout_->addItem(filter2d_smooth_high_);

    // Record spin box
    contrast_upper_threshold_ = new QDoubleSpinBoxLayout(parent, parent_widget, "Contrast_upper_threshold");
    contrast_upper_threshold_->setValue(99.5f);
    advanced_layout_->addItem(contrast_upper_threshold_);

    // Output spin box
    renorm_constant_ = new QIntSpinBoxLayout(parent, parent_widget, "Renorm_constant");
    renorm_constant_->setValue(5);
    advanced_layout_->addItem(renorm_constant_);

    // 3D cuts spin box
    cuts_contrast_p_offset_ = new QIntSpinBoxLayout(parent, parent_widget, "Cuts_contrast_p_offset");
    cuts_contrast_p_offset_->setValue(0);
    advanced_layout_->addItem(cuts_contrast_p_offset_);

    setLayout(advanced_layout_);
}

ASWPanelAdvanced::~ASWPanelAdvanced() {}

} // namespace holovibes::gui